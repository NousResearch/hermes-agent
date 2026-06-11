from __future__ import annotations

import logging
from typing import Any, Protocol

from .config import MonicaConfig, runtime_root
from .repo_manager import is_safe_git_branch_name
from .state import MonicaState

SUPPORTED_ROLLOUT_MODES = {"dry_run", "linear_only", "local_fix_only", "approved_pr"}
LINEAR_ROLLOUT_MODES = {"linear_only", "local_fix_only", "approved_pr"}
CODE_ROLLOUT_MODES = {"local_fix_only", "approved_pr"}
logger = logging.getLogger(__name__)


class MonicaLoopSkills(Protocol):
    def read_slack_thread(self, run: Any) -> dict[str, Any]:
        ...

    def infer_user_intent(self, run: Any, thread: dict[str, Any]) -> dict[str, Any]:
        ...

    def create_or_update_linear(
        self,
        run: Any,
        thread: dict[str, Any],
        intent: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def ask_fix_approval(self, run: Any, issue: dict[str, Any]) -> None:
        ...

    def run_internal_codex_worker(self, run: Any) -> dict[str, Any]:
        ...

    def run_verification(self, run: Any, worker_result: dict[str, Any]) -> dict[str, Any]:
        ...

    def run_proof(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def open_draft_pr(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class MonicaLoop:
    def __init__(self, *, config: MonicaConfig, state: MonicaState, skills: MonicaLoopSkills) -> None:
        self.config = config
        self.state = state
        self.skills = skills

    def run(self, run_id: str) -> None:
        run = self.state.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        try:
            self._run(run)
        except Exception as exc:
            self._mark_failed(run_id, exc)

    def _run(self, run: Any) -> None:
        if self.config.rollout_mode not in SUPPORTED_ROLLOUT_MODES:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason=f"unknown_rollout_mode: {self.config.rollout_mode}",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "Monica is not configured with a known rollout mode, so I stopped before taking action. "
                "Use one of: `dry_run`, `linear_only`, `local_fix_only`, `approved_pr`.",
            )
            return

        if self.config.rollout_mode in LINEAR_ROLLOUT_MODES and not self.config.loop.create_linear:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="linear_creation_disabled_in_rollout",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "Linear creation is disabled for this Monica rollout mode, so I stopped before taking action.",
            )
            return

        if run.status == "approved":
            self._run_approved_fix(run)
            return

        if run.status in {"proof_blocked", "proofing"}:
            self._resume_after_proof_blocked(run)
            return

        if run.status not in {"queued", "needs_clarification"}:
            return

        run = self.state.update_run(run.id, status="triaging")
        thread = self.skills.read_slack_thread(run)
        if self._is_cancelled(run.id):
            return
        intent = self.skills.infer_user_intent(run, thread)
        if self._is_cancelled(run.id):
            return

        if intent.get("needs_clarification"):
            needs_clarification = self.state.update_run(run.id, status="needs_clarification")
            self._log_run("needs_clarification", needs_clarification, stage="triaging")
            self._post_status(needs_clarification, self._clarification_text(intent))
            return

        if not intent.get("is_mobile_bug"):
            blocked = self.state.update_run(run.id, status="blocked", failure_reason="not_a_mobile_bug")
            self._log_run("blocked", blocked, stage="triaging")
            self._post_status(
                blocked,
                "I could not confidently classify this as a mobile app bug. "
                "Tag me again with the app/platform details if you want me to file it.",
            )
            return

        if intent.get("wants_linear") is False and not intent.get("wants_fix"):
            needs_clarification = self.state.update_run(run.id, status="needs_clarification")
            self._log_run("needs_clarification", needs_clarification, stage="triaging")
            self._post_status(
                needs_clarification,
                "I see mobile app bug context here, but I do not have a clear next step yet. "
                "Tag me again if you want me to file a Linear issue or prepare a fix.",
            )
            return

        if self.config.loop.create_linear:
            run = self.state.update_run(run.id, status="creating_linear")
            issue = self.skills.create_or_update_linear(run, thread, intent)
            if self._is_cancelled(run.id):
                return
        else:
            issue = {"identifier": "", "url": "", "dry_run": True, "title": intent.get("summary", "")}
        run = self.state.update_run(
            run.id,
            status="linear_created",
            linear_identifier=issue.get("identifier", ""),
            linear_issue_id=issue.get("id", ""),
            linear_url=issue.get("url", ""),
        )

        if issue.get("dry_run"):
            self._post_status(run, self._linear_done_text(issue))
            completed = self.state.update_run(run.id, status="done")
            self._log_run("done", completed, stage="dry_run")
            return

        if intent.get("wants_fix") and self.config.rollout_mode not in CODE_ROLLOUT_MODES:
            self._post_status(
                run,
                self._linear_done_text(issue)
                + "\n\nCode fixes are disabled in the current Monica rollout mode.",
            )
            completed = self.state.update_run(run.id, status="done")
            self._log_run("done", completed, stage=self.config.rollout_mode)
            return

        if intent.get("wants_fix"):
            run = self.state.update_run(run.id, status="awaiting_fix_approval")
            self._log_run("awaiting_fix_approval", run, stage="linear_created")
            self.skills.ask_fix_approval(run, issue)
            return

        self._post_status(run, self._linear_done_text(issue))
        completed = self.state.update_run(run.id, status="done")
        self._log_run("done", completed, stage=self.config.rollout_mode)

    def _run_approved_fix(self, run: Any) -> None:
        if self.config.rollout_mode not in CODE_ROLLOUT_MODES:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="approved_pr_rollout_not_enabled",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "I have approval, but code rollout is not enabled. "
                "Set `mobile_bug_agent.rollout_mode` to `local_fix_only` or `approved_pr` before I write code.",
            )
            return

        if not (run.linear_identifier or run.linear_issue_id or run.linear_url):
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="linear_issue_missing_before_fix",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "I have approval, but no Linear issue is attached to this Monica run, "
                "so I stopped before writing code.",
            )
            return

        if not any(command.strip() for command in self.config.verification.commands):
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="verification_commands_missing",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "I have approval, but `mobile_bug_agent.verification.commands` is empty, "
                "so I stopped before writing code.",
            )
            return

        run = self.state.update_run(run.id, status="fixing")
        worker_result = self.skills.run_internal_codex_worker(run)
        if self._is_cancelled(run.id):
            return
        branch_name = str(worker_result.get("branch_name") or run.branch_name or "")
        if not branch_name:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="worker_branch_missing",
            )
            self._log_run("blocked", blocked, stage="fixing")
            self._post_status(
                blocked,
                "I stopped after the code worker because I could not identify a branch to verify and push.",
            )
            return
        if not self._is_expected_worker_branch(run=run, branch_name=branch_name):
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="worker_branch_mismatch",
            )
            self._log_run("blocked", blocked, stage="fixing")
            self._post_status(
                blocked,
                "I stopped after the code worker because it returned an unexpected branch "
                f"`{branch_name}` for this Monica run.",
            )
            return
        run = self.state.update_run(run.id, branch_name=branch_name)
        if worker_result.get("changed") is False:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="worker_no_changes",
            )
            self._log_run("blocked", blocked, stage="fixing")
            self._post_status(
                blocked,
                "I stopped after the code worker because it did not report any code changes.",
            )
            return

        self._run_post_worker_gates(run, worker_result)

    def _resume_after_proof_blocked(self, run: Any) -> None:
        if self.config.rollout_mode not in CODE_ROLLOUT_MODES:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="proof_retry_rollout_not_enabled",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "I have a proof-blocked code branch, but code rollout is not enabled. "
                "Set `mobile_bug_agent.rollout_mode` to `local_fix_only` or `approved_pr` before retrying.",
            )
            return

        branch_name = str(getattr(run, "branch_name", "") or "").strip()
        if not branch_name:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="proof_retry_branch_missing",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "I cannot retry proof because this Monica run does not have a stored branch name.",
            )
            return
        if not is_safe_git_branch_name(branch_name) or not self._is_expected_worker_branch(
            run=run, branch_name=branch_name
        ):
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="proof_retry_branch_mismatch",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "I cannot retry proof because the stored branch does not look like the expected Monica branch "
                f"for this run: `{branch_name}`.",
            )
            return

        worktree_path = (
            runtime_root(self.config)
            / "workspace"
            / "worktrees"
            / branch_name.replace("/", "-")
        )
        if not worktree_path.is_dir() or not (worktree_path / ".git").exists():
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="proof_retry_worktree_missing",
            )
            self._log_run("blocked", blocked, stage="preflight")
            self._post_status(
                blocked,
                "I cannot retry proof because the stored Monica worktree is missing: "
                f"`{worktree_path}`.",
            )
            return

        worker_result = {
            "branch_name": branch_name,
            "worktree_path": str(worktree_path),
            "changed": True,
            "slack_permalink": self._run_permalink(run),
            "summary": "Resuming Monica from the existing proof-blocked branch.",
            "evidence": [],
        }
        self._run_post_worker_gates(run, worker_result)

    def _run_post_worker_gates(self, run: Any, worker_result: dict[str, Any]) -> None:
        run = self.state.update_run(run.id, status="verifying")
        verification = self.skills.run_verification(run, worker_result)
        if self._is_cancelled(run.id):
            return
        if not verification.get("passed"):
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason=f"verification_failed: {verification.get('summary', '')}".strip(),
            )
            self._log_run("blocked", blocked, stage="verifying")
            self._post_status(run, f"Verification failed, so I did not open a PR.\n{verification.get('summary', '')}")
            return

        if self._proof_required():
            run = self.state.update_run(run.id, status="proofing")
            proof = self.skills.run_proof(run, worker_result, verification)
            if self._is_cancelled(run.id):
                return
            artifacts = self._proof_artifacts(proof)
            if not proof.get("passed") or not artifacts:
                summary = str(proof.get("summary") or "Proof unavailable.").strip()
                blocked = self.state.update_run(
                    run.id,
                    status="proof_blocked",
                    failure_reason=f"proof_unavailable: {summary}".strip(),
                )
                self._log_run("proof_blocked", blocked, stage="proofing")
                self._post_status(
                    blocked,
                    "Verification passed, but simulator proof is unavailable, "
                    "so I did not mark this run done or open a PR.\n"
                    f"{summary}",
                )
                return
            worker_result["proof"] = dict(proof)

        if self.config.rollout_mode == "local_fix_only":
            completed = self.state.update_run(run.id, status="done")
            self._log_run("done", completed, stage="local_fix_only")
            proof_text = self._proof_status_text(worker_result.get("proof"))
            self._post_status(
                completed,
                "Local fix is ready on branch "
                f"`{completed.branch_name}`. Verification passed. "
                "The branch was not pushed and no PR was opened."
                f"{proof_text}",
            )
            return

        run = self.state.update_run(run.id, status="opening_pr")
        pr = self.skills.open_draft_pr(run, worker_result, verification)
        pr_url = str(pr.get("url") or "")
        if not pr_url:
            blocked = self.state.update_run(
                run.id,
                status="blocked",
                failure_reason="draft_pr_url_missing",
            )
            self._log_run("blocked", blocked, stage="opening_pr")
            self._post_status(
                blocked,
                "I opened the final PR stage, but the publisher did not return a draft PR URL, "
                "so I stopped before marking this run complete.",
            )
            return
        if self._is_cancelled(run.id):
            if pr_url:
                cancelled = self.state.update_run(run.id, pr_url=pr_url)
                self._log_run("blocked", cancelled, stage="opening_pr")
            return
        completed = self.state.update_run(
            run.id,
            status="done",
            pr_url=pr_url,
        )
        self._log_run("done", completed, stage="opening_pr")
        self._post_status(completed, f"Draft PR is ready: {pr_url}")

    @staticmethod
    def _run_permalink(run: Any) -> str:
        raw_event = getattr(run, "raw_event", None)
        if isinstance(raw_event, dict):
            return str(raw_event.get("permalink") or "").strip()
        return ""

    def _is_cancelled(self, run_id: str) -> bool:
        run = self.state.get_run(run_id)
        return bool(
            run
            and run.status == "blocked"
            and str(run.failure_reason or "").startswith("cancelled by ")
        )

    def _post_status(self, run: Any, text: str) -> None:
        poster = getattr(self.skills, "post_status", None)
        if callable(poster):
            try:
                poster(run, text)
            except Exception:
                return

    def _proof_required(self) -> bool:
        return bool(self.config.proof.required_for_done)

    @staticmethod
    def _proof_artifacts(proof: dict[str, Any]) -> list[str]:
        return [str(path).strip() for path in proof.get("artifacts") or [] if str(path).strip()]

    def _proof_status_text(self, proof: object) -> str:
        if not isinstance(proof, dict):
            return ""
        artifacts = self._proof_artifacts(proof)
        if not artifacts:
            return ""
        visible = ", ".join(artifacts[:5])
        suffix = "" if len(artifacts) <= 5 else f", +{len(artifacts) - 5} more"
        return f"\nProof captured: {visible}{suffix}"

    def _is_expected_worker_branch(self, *, run: Any, branch_name: str) -> bool:
        branch = str(branch_name or "").strip()
        if not branch:
            return False
        prefix = str(self.config.repo.branch_prefix or "").strip().rstrip("/")
        if prefix and not branch.startswith(f"{prefix}/"):
            return False
        linear_identifier = str(getattr(run, "linear_identifier", "") or "").strip()
        return not linear_identifier or linear_identifier in branch

    def _mark_failed(self, run_id: str, exc: Exception) -> None:
        run = self.state.get_run(run_id)
        if run is None:
            raise KeyError(run_id) from exc
        stage = run.status or "unknown"
        detail = str(exc) or exc.__class__.__name__
        failed = self.state.update_run(
            run_id,
            status="failed",
            failure_reason=f"{stage}_failed: {detail}",
        )
        self._log_run("failed", failed, stage=stage)
        self._post_status(
            failed,
            "I hit a problem while working this Monica run, so I stopped before taking the next action.\n"
            f"Stage: {stage}\n"
            f"Check Monica logs or `hermes mobile-bug-agent show {failed.id}` on the host for details.",
        )

    @staticmethod
    def _log_run(event: str, run: Any, *, stage: str = "") -> None:
        logger.info(
            "monica_run event=%s run_id=%s status=%s stage=%s channel_id=%s thread_ts=%s "
            "linear_identifier=%s linear_url=%s branch_name=%s pr_url=%s failure_reason=%s",
            event,
            getattr(run, "id", ""),
            getattr(run, "status", ""),
            stage,
            getattr(run, "channel_id", ""),
            getattr(run, "thread_ts", ""),
            getattr(run, "linear_identifier", ""),
            getattr(run, "linear_url", ""),
            getattr(run, "branch_name", ""),
            getattr(run, "pr_url", ""),
            getattr(run, "failure_reason", ""),
        )

    @staticmethod
    def _clarification_text(intent: dict[str, Any]) -> str:
        questions = intent.get("missing_questions") or []
        if isinstance(questions, str):
            questions = [questions]
        clean = [str(question).strip() for question in questions if str(question).strip()]
        if not clean:
            return "I need a little more context before I file this as a mobile bug."
        return "I need a little more context before I file this:\n" + "\n".join(f"- {q}" for q in clean)

    @staticmethod
    def _linear_done_text(issue: dict[str, Any]) -> str:
        if issue.get("dry_run"):
            preview = _compact(str(issue.get("description") or ""), limit=500)
            if preview:
                return (
                    f"Dry run: I would create a Linear issue titled `{issue.get('title', 'Mobile bug')}`.\n"
                    f"Preview:\n{preview}"
                )
            return f"Dry run: I would create a Linear issue titled `{issue.get('title', 'Mobile bug')}`."
        if issue.get("url"):
            return f"Created Linear issue: {issue.get('url')}"
        return "Created the Linear issue."


def _compact(value: str, *, limit: int) -> str:
    compacted = "\n".join(line.rstrip() for line in value.strip().splitlines() if line.strip())
    if len(compacted) <= limit:
        return compacted
    return compacted[: max(0, limit - 3)].rstrip() + "..."
