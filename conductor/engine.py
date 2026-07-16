from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import DEFAULT_CONDUCTOR_CONFIG
from .models import StepKind, WorkerState
from .receipts import verify_receipt


class TickResult(str, Enum):
    LAUNCHED_WRITER = "LAUNCHED_WRITER"
    LAUNCHED_REVIEWER = "LAUNCHED_REVIEWER"
    ADOPTED_PROGRESSING = "ADOPTED_PROGRESSING"
    WAITING_STALE = "WAITING_STALE"
    RETRY_BACKOFF = "RETRY_BACKOFF"
    BLOCKED_INVALID_RECEIPT = "BLOCKED_INVALID_RECEIPT"
    BLOCKED_BUDGET = "BLOCKED_BUDGET"
    BLOCKED_HUMAN = "BLOCKED_HUMAN"
    BUSY = "BUSY"
    OBSERVED_SILENT = "OBSERVED_SILENT"
    ADVANCED = "ADVANCED"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class LaunchSpec:
    worker_id: str
    campaign_id: str
    step_index: int
    role: str
    command: tuple[str, ...]
    cwd: str
    tmux_session: str
    provider: str
    model: str
    prompt_path: str
    prompt_hash: str
    mutable_manifest: tuple[str, ...]
    output_path: str
    receipt_path: str
    heartbeat_path: str
    nonce: str
    read_only: bool
    protected_roots: tuple[str, ...]


DEFAULT_BUDGETS = DEFAULT_CONDUCTOR_CONFIG["budgets"]


class Conductor:
    def __init__(
        self, store, launcher, *, now=None, observer=None, tick_lease_seconds=30
    ):
        self.store = store
        self.launcher = launcher
        self.now = now or time.time
        self.observer = observer
        if not 1 <= float(tick_lease_seconds) <= 3600:
            raise ValueError("tick_lease_seconds must be between 1 and 3600")
        self.tick_lease_seconds = float(tick_lease_seconds)

    def record_progress(self, worker_id: str, evidence: str) -> None:
        self.store.update_worker(
            worker_id, heartbeat_at=self.now(), progress_evidence=evidence
        )

    def _block(self, campaign_id: str, key: str) -> TickResult:
        self.store.update_campaign(campaign_id, state="BLOCKED", blocker_key=key)
        return TickResult.BLOCKED_BUDGET

    def tick(self, campaign_id: str) -> TickResult:
        owner = uuid.uuid4().hex
        if not self.store.acquire_tick(
            campaign_id, owner, lease_seconds=self.tick_lease_seconds, now=self.now()
        ):
            return TickResult.BUSY
        try:
            campaign = self.store.get_campaign(campaign_id)
            budgets = {**DEFAULT_BUDGETS, **campaign.plan.budgets}
            worker = self.store.active_worker(campaign_id)
            if (
                worker
                and worker.step_index == campaign.step_index
                and worker.state is WorkerState.RUNNING
            ):
                return self._poll_worker(campaign, worker, budgets)
            if campaign.state == "COMPLETE":
                return TickResult.COMPLETE
            if campaign.state == "BLOCKED" and worker:
                return TickResult.BLOCKED_INVALID_RECEIPT
            if self.now() < campaign.next_retry_at:
                return TickResult.RETRY_BACKOFF
            return self._start_step(campaign, budgets)
        finally:
            self.store.release_tick(campaign_id, owner)

    def _consume_turn(self, campaign, budgets) -> bool:
        turns = campaign.conductor_turns + 1
        if turns > int(budgets["max_conductor_turns"]):
            self._block(
                campaign.campaign_id,
                "max_conductor_turns: raise conductor.budgets.max_conductor_turns",
            )
            return False
        self.store.update_campaign(campaign.campaign_id, conductor_turns=turns)
        return True

    def _budget_allows(self, campaign, budgets) -> bool:
        usage = self.store.daily_usage(
            campaign.campaign_id, time.strftime("%Y-%m-%d", time.gmtime(self.now()))
        )
        if usage["processed_tokens"] >= int(budgets["max_processed_tokens_per_day"]):
            return False
        remaining = (
            int(budgets["max_processed_tokens_per_day"]) - usage["processed_tokens"]
        )
        if remaining < int(budgets["max_processed_tokens_per_run"]):
            return False
        if usage["runs"] >= int(budgets["max_runs_per_day"]):
            return False
        return True

    def _start_step(self, campaign, budgets) -> TickResult:
        if not self._consume_turn(campaign, budgets):
            return TickResult.BLOCKED_BUDGET
        if campaign.step_index >= len(campaign.plan.steps):
            self.store.update_campaign(campaign.campaign_id, state="COMPLETE")
            return TickResult.COMPLETE
        step = campaign.plan.steps[campaign.step_index]
        if step.kind is StepKind.HUMAN_DECISION:
            self.store.update_campaign(
                campaign.campaign_id,
                state="BLOCKED",
                blocker_key=f"human:{step.step_id}",
            )
            return TickResult.BLOCKED_HUMAN
        if step.kind in (StepKind.DETERMINISTIC_GATE, StepKind.OBSERVATION):
            completed = subprocess.run(
                step.command,
                cwd=campaign.plan.cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                self.store.update_campaign(
                    campaign.campaign_id,
                    state="BLOCKED",
                    blocker_key=f"command:{step.step_id}",
                )
                return TickResult.BLOCKED_INVALID_RECEIPT
            self.store.update_campaign(
                campaign.campaign_id, step_index=campaign.step_index + 1, state="READY"
            )
            if step.kind is StepKind.OBSERVATION and not completed.stdout.strip():
                return TickResult.OBSERVED_SILENT
            return TickResult.ADVANCED
        if not self._budget_allows(campaign, budgets):
            return self._block(
                campaign.campaign_id,
                "daily budget exhausted: retry after 00:00 UTC or raise conductor.budgets",
            )

        role = "writer" if step.kind is StepKind.IMPLEMENTATION else "reviewer"
        routing = campaign.plan.writer if role == "writer" else campaign.plan.reviewer
        command = routing.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            return self._block(
                campaign.campaign_id, f"missing configured {role} command"
            )
        worker_id = (
            f"{campaign.campaign_id}-{campaign.step_index}-{uuid.uuid4().hex[:10]}"
        )
        root = self.store.path.parent / "runs" / worker_id
        root.mkdir(parents=True, exist_ok=False)
        prompt = json.dumps(
            {
                "campaign": campaign.campaign_id,
                "step": step.step_id,
                "role": role,
                "instructions": step.prompt,
                "mutable_manifest": campaign.plan.mutable_manifest,
            },
            sort_keys=True,
        )
        prompt_path = root / "prompt.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        tmux_session = (
            f"hermes-cond-{hashlib.sha256(worker_id.encode()).hexdigest()[:18]}"
        )
        spec = LaunchSpec(
            worker_id=worker_id,
            campaign_id=campaign.campaign_id,
            step_index=campaign.step_index,
            role=role,
            command=tuple(command),
            cwd=str(Path(campaign.plan.cwd).resolve()),
            tmux_session=tmux_session,
            provider=str(routing.get("provider", "")),
            model=str(routing.get("model", "")),
            prompt_path=str(prompt_path),
            prompt_hash=prompt_hash,
            mutable_manifest=tuple(campaign.plan.mutable_manifest),
            output_path=str(root / "output.json"),
            receipt_path=str(root / "receipt.json"),
            heartbeat_path=str(root / "heartbeat.json"),
            nonce=uuid.uuid4().hex,
            read_only=(role == "reviewer"),
            protected_roots=self._protected_roots(campaign.plan.cwd)
            if role == "reviewer"
            else (),
        )
        self.store.insert_worker({
            "worker_id": spec.worker_id,
            "campaign_id": spec.campaign_id,
            "step_index": campaign.step_index,
            "role": spec.role,
            "cwd": spec.cwd,
            "tmux_session": spec.tmux_session,
            "pid": None,
            "start_marker": None,
            "provider": spec.provider,
            "model": spec.model,
            "prompt_hash": spec.prompt_hash,
            "manifest_json": json.dumps(spec.mutable_manifest),
            "launched_at": self.now(),
            "heartbeat_at": None,
            "progress_evidence": None,
            "state": WorkerState.RUNNING.value,
            "output_path": spec.output_path,
            "receipt_path": spec.receipt_path,
            "receipt_hash": None,
            "nonce": spec.nonce,
            "read_only": int(spec.read_only),
        })
        # Durable WAITING transition happens before process launch. A launch failure
        # therefore remains recoverable state and can never turn into conductor takeover.
        waiting = "WAITING_WRITER" if role == "writer" else "WAITING_REVIEW"
        self.store.update_campaign(campaign.campaign_id, state=waiting)
        metadata = self.launcher.launch(spec)
        self.store.update_worker(
            spec.worker_id,
            pid=metadata.get("pid"),
            start_marker=metadata.get("start_marker"),
        )
        return (
            TickResult.LAUNCHED_WRITER
            if role == "writer"
            else TickResult.LAUNCHED_REVIEWER
        )

    def _poll_worker(self, campaign, worker, budgets) -> TickResult:
        running = self.launcher.is_running(worker.tmux_session)
        receipt_path = Path(worker.receipt_path)
        if running:
            if self.now() - worker.launched_at > float(budgets["wall_time_seconds"]):
                role_state = (
                    "WAITING_WRITER" if worker.role == "writer" else "WAITING_REVIEW"
                )
                self.store.update_campaign(
                    campaign.campaign_id,
                    state=role_state,
                    blocker_key=(
                        "worker wall time exceeded: inspect the tracked session "
                        "and provide a terminal receipt"
                    ),
                )
            if (
                worker.heartbeat_at is not None
                and worker.heartbeat_at >= worker.launched_at
                and worker.progress_evidence
            ):
                return TickResult.ADOPTED_PROGRESSING
            return TickResult.WAITING_STALE
        if receipt_path.exists():
            if not self._consume_turn(campaign, budgets):
                return TickResult.BLOCKED_BUDGET
            value, error = verify_receipt(
                receipt_path, worker, int(budgets["max_worker_turns"])
            )
            if error:
                self.store.update_campaign(
                    campaign.campaign_id, state="BLOCKED", blocker_key=error
                )
                return TickResult.BLOCKED_INVALID_RECEIPT
            usage = value["usage"]
            processed = sum(
                max(0, int(usage.get(key, 0) or 0))
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "cache_read_tokens",
                )
            )
            if processed > int(budgets["max_processed_tokens_per_run"]):
                return self._block(
                    campaign.campaign_id,
                    "run token budget exceeded: lower worker scope or raise limit",
                )
            day = time.strftime("%Y-%m-%d", time.gmtime(self.now()))
            self.store.add_usage(campaign.campaign_id, usage, day)
            self.store.update_worker(
                worker.worker_id,
                state=WorkerState.VERIFIED,
                receipt_hash=value["receipt_hash"],
            )
            self.launcher.cleanup(worker.tmux_session)
            next_step = campaign.step_index + 1
            state = "COMPLETE" if next_step >= len(campaign.plan.steps) else "READY"
            self.store.update_campaign(
                campaign.campaign_id, step_index=next_step, state=state, retries=0
            )
            return TickResult.COMPLETE if state == "COMPLETE" else TickResult.ADVANCED

        if self.now() < campaign.next_retry_at:
            return TickResult.RETRY_BACKOFF
        if not self._consume_turn(campaign, budgets):
            return TickResult.BLOCKED_BUDGET
        retries = campaign.retries + 1
        if retries > int(budgets["max_retries"]):
            self.store.update_campaign(
                campaign.campaign_id,
                state="BLOCKED",
                blocker_key="worker exited without receipt",
            )
            return TickResult.BLOCKED_INVALID_RECEIPT
        backoff = float(budgets["backoff_base_seconds"]) * (2 ** (retries - 1))
        retry_state = "WAITING_WRITER" if worker.role == "writer" else "WAITING_REVIEW"
        self.store.update_campaign(
            campaign.campaign_id,
            retries=retries,
            next_retry_at=self.now() + backoff,
            state=retry_state,
        )
        return TickResult.RETRY_BACKOFF

    @staticmethod
    def _protected_roots(cwd: str) -> tuple[str, ...]:
        roots = [str(Path(cwd).resolve())]
        for flag in ("--absolute-git-dir", "--git-common-dir"):
            result = subprocess.run(
                ["git", "-C", cwd, "rev-parse", flag],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                continue
            path = Path(result.stdout.strip())
            if not path.is_absolute():
                path = Path(cwd) / path
            resolved = str(path.resolve())
            if resolved not in roots:
                roots.append(resolved)
        return tuple(roots)
