"""DevRun/DevFlow slash command handlers for GatewayRunner."""

from __future__ import annotations

import asyncio

from gateway.platforms.base import MessageEvent


class GatewayDevRunCommandsMixin:
    async def _handle_devrun_command(self, event: MessageEvent) -> str:
        """Handle /devrun — start a cautious mobile development job."""
        from gateway.devrun import (
            DevRunJob,
            HIGH_RISK,
            build_execution_prompt,
            classify_risk,
            find_sidecar_command,
            make_job_id,
            parse_devrun_args,
            resolve_repo_path,
            save_job,
        )

        raw_args = event.get_command_args().strip()
        try:
            request = parse_devrun_args(raw_args)
            repo = resolve_repo_path(request.repo)
        except ValueError as exc:
            return str(exc)

        risk = classify_risk(request.task, repo)
        job_id = make_job_id()
        sidecar_available = find_sidecar_command() is not None
        status = "queued" if risk == "low" else "awaiting_approval" if risk == HIGH_RISK else "reviewing" if sidecar_available else "blocked"
        review_summary = ""
        if risk == HIGH_RISK:
            review_summary = "High-risk DevRun requires mobile approval before review or execution."
        elif risk != "low" and not sidecar_available:
            review_summary = (
                "OpenSquilla sidecar is not available on this host; "
                "non-low-risk DevRun was blocked before execution."
            )

        job = DevRunJob(
            job_id=job_id,
            raw_task=raw_args,
            task=request.task,
            repo=str(repo),
            risk=risk,
            status=status,
            review_summary=review_summary,
        )
        save_job(job)

        source = event.source
        event_message_id = self._reply_anchor_for_event(event)

        if risk == "low":
            prompt = build_execution_prompt(job)
            _task = asyncio.create_task(
                self._run_devrun_background_task(
                    job_id=job_id,
                    prompt=prompt,
                    source=source,
                    event_message_id=event_message_id,
                )
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
        elif risk == HIGH_RISK:
            async def _on_devrun_confirm(choice: str) -> str:
                from gateway.devrun import load_job, save_job, summarize_job

                confirmed_job = load_job(job_id)
                if choice == "cancel":
                    confirmed_job.cancelled = True
                    confirmed_job.status = "blocked"
                    confirmed_job.error = "Cancelled by user before execution."
                    confirmed_job.telegram_messages.append("High-risk DevRun cancelled by user.")
                    save_job(confirmed_job)
                    return "DevRun cancelled.\n" + summarize_job(confirmed_job)

                if find_sidecar_command() is None:
                    confirmed_job.status = "blocked"
                    confirmed_job.error = "OpenSquilla sidecar unavailable; high-risk DevRun was not executed."
                    confirmed_job.review_summary = (
                        confirmed_job.review_summary
                        or "OpenSquilla sidecar unavailable."
                    )
                    save_job(confirmed_job)
                    return "DevRun not executed.\n" + summarize_job(confirmed_job)

                confirmed_job.status = "reviewing"
                confirmed_job.telegram_messages.append(
                    "High-risk DevRun approved by user; OpenSquilla review starting."
                )
                save_job(confirmed_job)
                _task = asyncio.create_task(
                    self._run_devrun_review_then_execute(
                        job_id=job_id,
                        source=source,
                        event_message_id=event_message_id,
                    )
                )
                self._background_tasks.add(_task)
                _task.add_done_callback(self._background_tasks.discard)
                return (
                    f"DevRun approved: {job_id}\n"
                    "OpenSquilla review started.\n"
                    f"Use /devstatus {job_id} to check progress."
                )

            _p = self._typed_command_prefix_for(event.source.platform)
            return await self._request_slash_confirm(
                event=event,
                command="devrun",
                title="High-Risk DevRun Confirmation",
                message=(
                    "High-risk DevRun needs mobile approval before anything runs.\n\n"
                    f"Job: {job_id}\n"
                    f"Repo: {repo}\n"
                    f"Task: {request.task}\n\n"
                    "Approve Once starts OpenSquilla review; execution starts only if review returns PASS. "
                    "Always Approve is treated as approve-once for DevRun.\n\n"
                    f"Text fallback: reply `{_p}approve` or `{_p}cancel`."
                ),
                handler=_on_devrun_confirm,
            )
        elif sidecar_available:
            _task = asyncio.create_task(
                self._run_devrun_review_then_execute(
                    job_id=job_id,
                    source=source,
                    event_message_id=event_message_id,
                )
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)

        lines = [
            f"DevRun job started: {job_id}",
            f"Status: {status}",
            f"Risk: {risk}",
            f"Repo: {repo}",
        ]
        if review_summary:
            lines.append(review_summary)
            if status == "blocked":
                lines.append("No background execution was started.")
        lines.append(f"Use /devstatus {job_id} to check progress.")
        return "\n".join(lines)

    async def _handle_devreview_command(self, event: MessageEvent) -> str:
        """Handle /devreview - run the OpenSquilla review council only."""
        from gateway.devrun import (
            DevRunJob,
            classify_risk,
            find_sidecar_command,
            make_job_id,
            parse_devrun_args,
            resolve_repo_path,
            save_job,
        )

        raw_args = event.get_command_args().strip()
        try:
            request = parse_devrun_args(raw_args)
            repo = resolve_repo_path(request.repo)
        except ValueError as exc:
            return str(exc)

        risk = classify_risk(request.task, repo)
        job_id = make_job_id()
        sidecar_available = find_sidecar_command() is not None
        status = "reviewing" if sidecar_available else "blocked"
        review_summary = (
            "DevReview council queued."
            if sidecar_available
            else "OpenSquilla sidecar is not available on this host; DevReview was not started."
        )

        job = DevRunJob(
            job_id=job_id,
            raw_task=raw_args,
            task=request.task,
            repo=str(repo),
            risk=risk,
            status=status,
            review_summary=review_summary,
        )
        if not sidecar_available:
            job.review_verdict = "UNAVAILABLE"
            job.error = review_summary
        save_job(job)

        if sidecar_available:
            source = event.source
            event_message_id = self._reply_anchor_for_event(event)
            _task = asyncio.create_task(
                self._run_devreview_background_task(
                    job_id=job_id,
                    source=source,
                    event_message_id=event_message_id,
                )
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)

        lines = [
            f"DevReview job started: {job_id}",
            f"Status: {status}",
            f"Risk: {risk}",
            f"Repo: {repo}",
            "Lenses: requirements, architecture, testing, security, final acceptance",
        ]
        if not sidecar_available:
            lines.append("No background review was started.")
        lines.append(f"Use /devstatus {job_id} to check progress.")
        return "\n".join(lines)

    async def _handle_devflow_command(self, event: MessageEvent) -> str:
        """Handle /devflow - one mobile entry for review -> route -> action."""
        from gateway.devrun import (
            DevRunJob,
            build_execution_prompt,
            classify_devflow_route,
            classify_risk,
            estimate_devflow_budget,
            format_devflow_budget_estimate,
            find_sidecar_command,
            infer_recent_repo_from_jobs,
            make_job_id,
            parse_devrun_args,
            render_devflow_budget_prompt,
            render_devflow_cancelled_card,
            render_devflow_started_card,
            resolve_kanban_board,
            resolve_repo_path,
            save_job,
        )

        raw_args = event.get_command_args().strip()
        try:
            request = parse_devrun_args(raw_args)
            board = resolve_kanban_board(request.board)
            if request.repo:
                repo = resolve_repo_path(request.repo)
                repo_source = "explicit"
            else:
                repo = infer_recent_repo_from_jobs()
                if repo is None:
                    return "\n".join(
                        [
                            "我还不知道要在哪个项目里做。",
                            "",
                            "第一次请这样发：",
                            "/devflow repo=/Users/yuxiansheng/dev/你的项目 task=我想做一个登录功能",
                            "",
                            "以后我会记住最近项目，你就可以直接发：",
                            "/devflow 我想做一个登录功能",
                        ]
                    )
                repo_source = "recent project"
        except ValueError as exc:
            return str(exc).replace("/devrun", "/devflow")

        risk = classify_risk(request.task, repo)
        route = classify_devflow_route(request.task, risk)
        sidecar_available = find_sidecar_command() is not None
        source = event.source
        event_message_id = self._reply_anchor_for_event(event)

        if route == "devrun":
            job_id = make_job_id()
            job = DevRunJob(
                job_id=job_id,
                raw_task=raw_args,
                task=request.task,
                repo=str(repo),
                risk=risk,
                status="queued",
                review_summary=(
                    "DevFlow route: low-risk task -> DevRun executor. "
                    "OpenSquilla review skipped for speed."
                ),
                board=board,
            )
            save_job(job)
            _task = asyncio.create_task(
                self._run_devrun_background_task(
                    job_id=job_id,
                    prompt=build_execution_prompt(job),
                    source=source,
                    event_message_id=event_message_id,
                )
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            return "\n".join(
                [
                    "🧭 **DevFlow 已接管**",
                    "",
                    "这是一个低风险小任务，我会直接交给 DevRun 执行，不启动 Kanban 军团。",
                    "",
                    f"任务：{request.task}",
                    f"项目：`{repo}`",
                    f"来源：{'最近项目' if repo_source == 'recent project' else '本次指定'}",
                    *( [f"Kanban：`{board}`"] if board else [] ),
                    f"风险：{'低风险' if risk == 'low' else risk}",
                    "",
                    f"进度查询：`/devstatus {job_id}`",
                ]
            )

        job_id = make_job_id()
        estimate = estimate_devflow_budget(
            request.task,
            risk,
            route,
            sidecar_available=sidecar_available,
        )
        budget_text = format_devflow_budget_estimate(estimate)
        review_summary = (
            "DevFlow route: Kanban triage. Awaiting mobile budget approval."
            if sidecar_available
            else "OpenSquilla sidecar unavailable; awaiting approval to create a Kanban triage card with review pending."
        )
        job = DevRunJob(
            job_id=job_id,
            raw_task=raw_args,
            task=request.task,
            repo=str(repo),
            risk=risk,
            status="awaiting_approval",
            review_summary=review_summary + "\n" + budget_text,
            board=board,
        )
        save_job(job)

        async def _on_devflow_budget_confirm(choice: str) -> str:
            from gateway.devrun import load_job, render_devflow_cancelled_card, render_devflow_started_card, save_job

            confirmed_job = load_job(job_id)
            if choice == "cancel":
                confirmed_job.cancelled = True
                confirmed_job.status = "blocked"
                confirmed_job.error = "Cancelled by user before DevFlow budget was spent."
                confirmed_job.telegram_messages.append("DevFlow budget gate cancelled by user.")
                save_job(confirmed_job)
                return render_devflow_cancelled_card(confirmed_job)

            confirmed_job.status = "reviewing" if sidecar_available else "queued"
            confirmed_job.review_summary = (
                "DevFlow route: Kanban triage. DevReview council queued.\n"
                + budget_text
                if sidecar_available
                else "OpenSquilla sidecar unavailable; DevFlow will create a Kanban triage card with review pending.\n"
                + budget_text
            )
            confirmed_job.telegram_messages.append("DevFlow budget approved by user.")
            save_job(confirmed_job)
            _task = asyncio.create_task(
                self._run_devflow_background_task(
                    job_id=job_id,
                    source=source,
                    event_message_id=event_message_id,
                )
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            return render_devflow_started_card(job=confirmed_job, repo_source=repo_source)

        _p = self._typed_command_prefix_for(event.source.platform)
        return await self._request_slash_confirm(
            event=event,
            command="devflow",
            title="DevFlow 智能开发流",
            message=render_devflow_budget_prompt(
                job_id=job_id,
                task=request.task,
                repo=repo,
                risk=risk,
                repo_source=repo_source,
                board=board,
                estimate=estimate,
                text_prefix=_p,
            ),
            handler=_on_devflow_budget_confirm,
        )

    async def _handle_devstatus_command(self, event: MessageEvent) -> str:
        """Handle /devstatus [job_id]."""
        from gateway.devrun import list_jobs, load_job, render_jobs, summarize_job

        job_id = event.get_command_args().strip()
        if job_id:
            try:
                return summarize_job(load_job(job_id))
            except KeyError:
                return f"DevRun job not found: {job_id}"
        return render_jobs(list_jobs(limit=10))

    async def _handle_devcancel_command(self, event: MessageEvent) -> str:
        """Handle /devcancel <job_id>."""
        from gateway.devrun import cancel_job, summarize_job

        job_id = event.get_command_args().strip()
        if not job_id:
            return "Usage: /devcancel <job_id>"
        try:
            job = cancel_job(job_id)
        except KeyError:
            return f"DevRun job not found: {job_id}"
        return "DevRun cancel requested.\n" + summarize_job(job)
