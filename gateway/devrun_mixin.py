"""DevRun/DevFlow background helpers for GatewayRunner."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from typing import Optional

from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayDevRunMixin:
    async def _run_devrun_background_task(
        self,
        *,
        job_id: str,
        prompt: str,
        source: "SessionSource",
        event_message_id: Optional[str] = None,
    ) -> None:
        """Run a DevRun job through the existing background-agent path."""
        from gateway.devrun import collect_new_repo_changes, load_job, record_repo_baseline, save_job, summarize_job

        try:
            job = load_job(job_id)
            if job.cancelled:
                return
            record_repo_baseline(job)
            job.status = "running"
            job.telegram_messages.append("Background execution started.")
            save_job(job)
        except Exception:
            logger.debug("Could not mark DevRun job %s as running", job_id, exc_info=True)
            return

        try:
            await self._run_background_task(
                prompt=prompt,
                source=source,
                task_id=job_id,
                event_message_id=event_message_id,
            )
        except Exception as exc:
            logger.exception("DevRun background task %s failed", job_id)
            try:
                job = load_job(job_id)
                job.status = "failed"
                job.error = f"Background task failed: {exc}"
                save_job(job)
            except Exception:
                logger.debug("Could not mark DevRun job %s as failed", job_id, exc_info=True)
            await self._send_devrun_status_update(source, event_message_id, job_id)
            return

        try:
            job = load_job(job_id)
            if job.cancelled:
                job.status = "blocked"
            elif job.status not in {"failed", "blocked"}:
                job.status = "done"
                job.telegram_messages.append("Background execution finished.")
            changed_files, change_summary = collect_new_repo_changes(job)
            job.changed_files = changed_files
            if change_summary and not job.test_results:
                job.test_results = change_summary
            save_job(job)
        except Exception:
            logger.debug("Could not finalize DevRun job %s", job_id, exc_info=True)

        adapter = self.adapters.get(source.platform)
        if not adapter:
            return
        try:
            job = load_job(job_id)
            metadata = self._thread_metadata_for_source(source, event_message_id)
            await adapter.send(
                source.chat_id,
                "DevRun status update\n" + summarize_job(job),
                metadata=metadata,
            )
        except Exception:
            logger.debug("Could not send DevRun status update for %s", job_id, exc_info=True)


    async def _run_devrun_review_then_execute(
        self,
        *,
        job_id: str,
        source: "SessionSource",
        event_message_id: Optional[str] = None,
    ) -> None:
        """Run OpenSquilla review before executing a non-low-risk DevRun job."""
        from gateway.devrun import (
            build_execution_prompt,
            load_job,
            run_sidecar_review,
            save_job,
            summarize_job,
        )

        try:
            job = load_job(job_id)
            if job.cancelled:
                return
            job.status = "reviewing"
            save_job(job)
            verdict, summary = await self._run_in_executor_with_context(
                run_sidecar_review,
                job,
            )
            job = load_job(job_id)
            job.review_verdict = verdict
            job.review_summary = summary
            if job.cancelled:
                job.status = "blocked"
                save_job(job)
                return
            if verdict != "PASS":
                job.status = "blocked"
                job.error = "OpenSquilla review did not return PASS."
                save_job(job)
                await self._send_devrun_status_update(source, event_message_id, job_id)
                return
            if job.status != "reviewing":
                job.status = "blocked"
                job.error = "DevRun state changed before execution; execution was not started."
                save_job(job)
                await self._send_devrun_status_update(source, event_message_id, job_id)
                return
            job.status = "queued"
            save_job(job)
        except Exception as exc:
            logger.exception("DevRun review failed for %s", job_id)
            try:
                job = load_job(job_id)
                job.status = "blocked"
                job.error = f"Review failed: {exc}"
                save_job(job)
            except Exception:
                pass
            await self._send_devrun_status_update(source, event_message_id, job_id)
            return

        await self._run_devrun_background_task(
            job_id=job_id,
            prompt=build_execution_prompt(job),
            source=source,
            event_message_id=event_message_id,
        )


    async def _run_devreview_background_task(
        self,
        *,
        job_id: str,
        source: "SessionSource",
        event_message_id: Optional[str] = None,
    ) -> None:
        """Run the OpenSquilla DevReview council without executing code changes."""
        from gateway.devrun import devreview_report_failed, load_job, run_devreview_council, save_job

        try:
            job = load_job(job_id)
            if job.cancelled:
                return
            job.status = "reviewing"
            job.telegram_messages.append("DevReview council started.")
            save_job(job)

            report = await self._run_in_executor_with_context(
                run_devreview_council,
                job,
            )

            job = load_job(job_id)
            if job.cancelled:
                job.status = "blocked"
                job.error = "Cancelled by user during DevReview."
            elif devreview_report_failed(report):
                job.status = "blocked"
                job.review_verdict = "BLOCKED"
                job.review_summary = report
                job.error = "OpenSquilla DevReview workers failed before producing review reports."
                job.test_results = "review-only; no tests run"
            else:
                job.status = "done"
                job.review_verdict = "DEVREVIEW"
                job.review_summary = report
                job.test_results = "review-only; no tests run"
                job.changed_files = []
                job.telegram_messages.append("DevReview council finished.")
            save_job(job)
        except Exception as exc:
            logger.exception("DevReview council failed for %s", job_id)
            try:
                job = load_job(job_id)
                job.status = "failed"
                job.error = f"DevReview failed: {exc}"
                save_job(job)
            except Exception:
                pass

        await self._send_devrun_status_update(source, event_message_id, job_id)


    async def _run_devflow_background_task(
        self,
        *,
        job_id: str,
        source: "SessionSource",
        event_message_id: Optional[str] = None,
    ) -> None:
        """Run DevFlow review and create a Kanban triage card when safe."""
        from gateway.devrun import (
            devreview_report_failed,
            estimate_devflow_budget,
            find_sidecar_command,
            format_devflow_budget_audit,
            load_job,
            run_devflow_preflight,
            save_job,
        )
        from hermes_cli.kanban import run_slash

        try:
            job = load_job(job_id)
            if job.cancelled:
                return
            sidecar_command = find_sidecar_command()
            sidecar_available = sidecar_command is not None
            estimate = estimate_devflow_budget(
                job.task,
                job.risk,
                "kanban",
                sidecar_available=sidecar_available,
            )
            budget_audit = format_devflow_budget_audit(estimate)
            report = ""
            review_status = "OpenSquilla review was not available; Kanban card requires human/specifier review."
            if sidecar_available:
                job.status = "reviewing"
                job.telegram_messages.append("DevFlow review started.")
                save_job(job)
                report = await self._run_in_executor_with_context(
                    run_devflow_preflight,
                    job,
                )
                review_status = (
                    "OpenSquilla review failed; Kanban card requires human/specifier review."
                    if devreview_report_failed(report)
                    else "OpenSquilla review completed."
                )
                job = load_job(job_id)
                if job.cancelled:
                    job.status = "blocked"
                    job.error = "Cancelled by user during DevFlow."
                    save_job(job)
                    await self._send_devrun_status_update(source, event_message_id, job_id)
                    return

            job.review_summary = "\n\n".join(
                part for part in (budget_audit, report or review_status) if part
            )
            job.test_results = "review-only before Kanban triage; no tests run"
            save_job(job)

            title = f"DevFlow: {job.task[:80]}"
            body = "\n\n".join(
                [
                    "Created by /devflow.",
                    f"Repo: {job.repo}",
                    f"Risk: {job.risk}",
                    "Route: DevReview -> Kanban triage.",
                    budget_audit,
                    "Current live verification scope:",
                    "\n".join(
                        [
                            "- Treat this Kanban root card body, child task events, and worker handoffs as the authoritative evidence for this run.",
                            "- DevReview findings are preflight questions to verify; they are not final blockers once live Kanban evidence answers them.",
                            "- Historical closure or evidence files only count when they explicitly match this job/card id.",
                        ]
                    ),
                    f"Review status: {review_status}",
                    "Original task:",
                    job.task,
                    "DevReview summary:",
                    (report or review_status)[:3000],
                ]
            )
            slash = " ".join(
                [
                    *( ["--board", shlex.quote(job.board)] if job.board else [] ),
                    "create",
                    shlex.quote(title),
                    "--body",
                    shlex.quote(body),
                    "--triage",
                    "--goal",
                    "--goal-max-turns",
                    "3",
                    "--workspace",
                    shlex.quote(f"dir:{job.repo}"),
                    "--created-by",
                    "devflow",
                    "--idempotency-key",
                    shlex.quote(f"devflow:{job.job_id}"),
                ]
            )
            output = await asyncio.to_thread(run_slash, slash)
            task_id_match = re.search(r"Created\s+(t_[0-9a-f]+)\b", output or "")
            if task_id_match:
                task_id = task_id_match.group(1)
                try:
                    platform = getattr(source, "platform", None)
                    platform_str = (
                        platform.value if hasattr(platform, "value") else str(platform or "")
                    ).lower()
                    chat_id = str(getattr(source, "chat_id", "") or "")
                    thread_id = str(getattr(source, "thread_id", "") or "")
                    user_id = str(getattr(source, "user_id", "") or "") or None
                    if platform_str and chat_id:
                        def _sub():
                            from hermes_cli import kanban_db as _kb
                            conn = _kb.connect(board=job.board)
                            try:
                                notifier_profile = getattr(self, "_kanban_notifier_profile", None)
                                if not notifier_profile and hasattr(self, "_active_profile_name"):
                                    notifier_profile = self._active_profile_name()
                                _kb.add_notify_sub(
                                    conn,
                                    task_id=task_id,
                                    platform=platform_str,
                                    chat_id=chat_id,
                                    thread_id=thread_id or None,
                                    user_id=user_id,
                                    notifier_profile=notifier_profile or "default",
                                )
                            finally:
                                conn.close()

                        await asyncio.to_thread(_sub)
                        output = (
                            (output or "").rstrip()
                            + "\nSubscribed to Kanban updates for "
                            + task_id
                            + "."
                        )
                except Exception as exc:
                    logger.warning("DevFlow kanban auto-subscribe failed: %s", exc)
            job = load_job(job_id)
            job.status = "done"
            job.review_verdict = "DEVFLOW"
            job.commands.append(f"kanban {slash}")
            job.telegram_messages.append("DevFlow created a Kanban triage card.")
            job.test_results = (output or "Kanban create returned no output.")[:1200]
            save_job(job)
        except Exception as exc:
            logger.exception("DevFlow failed for %s", job_id)
            try:
                job = load_job(job_id)
                job.status = "failed"
                job.error = f"DevFlow failed: {exc}"
                save_job(job)
            except Exception:
                pass

        await self._send_devrun_status_update(source, event_message_id, job_id)


    async def _send_devrun_status_update(
        self,
        source: "SessionSource",
        event_message_id: Optional[str],
        job_id: str,
    ) -> None:
        from gateway.devrun import load_job, summarize_job

        adapter = self.adapters.get(source.platform)
        if not adapter:
            return
        try:
            job = load_job(job_id)
            metadata = self._thread_metadata_for_source(source, event_message_id)
            if job.review_verdict == "DEVREVIEW":
                prefix = "DevReview status update"
            elif job.review_verdict == "DEVFLOW" or (job.review_summary or "").startswith("DevFlow route:"):
                prefix = "DevFlow status update"
            else:
                prefix = "DevRun status update"
            await adapter.send(
                source.chat_id,
                prefix + "\n" + summarize_job(job),
                metadata=metadata,
            )
        except Exception:
            logger.debug("Could not send DevRun status update for %s", job_id, exc_info=True)
