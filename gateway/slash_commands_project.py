"""`/new -project`: open the Desktop Project picker and commit the chosen workspace as a fresh session."""

from __future__ import annotations

import asyncio
import logging
import secrets
from pathlib import Path
from typing import Optional

from gateway.platforms.event import MessageEvent
from gateway.project_picker import (
    eligible_projects, list_projects_for_profile, new_request_for_event, picker_page,
    project_copy, validate_workspace,
)
from gateway.project_session import ProjectWorkspaceError, require_local_backend, route_transition_lock

logger = logging.getLogger(__name__)


class GatewayProjectPickerMixin:
    """Picker open / navigation / commit. Opening and navigating never touch the conversation."""

    async def _handle_reset_command_serialized(self, event: MessageEvent):
        """Ordinary `/new` under the same per-route lock as the picker commit, so the two can never
        stop each other's (or a freshly started) turn."""
        route = self._session_key_for_source(event.source)
        async with route_transition_lock(self, route):
            return await self._handle_reset_command(event)

    async def _handle_project_picker_command(self, event: MessageEvent) -> str | None:
        """Send the picker; the reply text (or None when the picker itself is the reply)."""
        source = event.source
        # Same restriction as plain `/new`: the topic-mode General lobby has no session to replace.
        if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
            return self._telegram_topic_root_new_message()
        if new_request_for_event(event).kind != "project":
            return project_copy("usage")
        adapter = self._delivery_adapter_for(source)
        if adapter is None or not callable(getattr(type(adapter), "send_project_picker", None)):
            return project_copy("unsupported")
        async with self._async_profile_scope_for_source(source):
            try:
                require_local_backend()
            except ProjectWorkspaceError as exc:
                return str(exc)
            profile_home = self._resolve_profile_home_for_source(source)
            try:
                projects = await asyncio.to_thread(list_projects_for_profile, profile_home)
            except Exception:
                logger.exception("Could not read the routed profile's Desktop Project registry")
                return project_copy("open_failed")
        eligible = eligible_projects(projects)
        session_key = self._session_key_for_source(source)
        token = secrets.token_urlsafe(9)
        state = {
            "source": source, "session_key": session_key,
            "expected_session_id": self.session_store.peek_session_id(session_key),
            "profile_home": str(profile_home), "projects": eligible, "project_id": None,
            "folders": [], "page": 0, "view": "projects", "chat_id": str(source.chat_id),
            "thread_id": str(source.thread_id or ""), "user_id": str(source.user_id or ""),
            "handler": lambda query, action, current: self._handle_project_picker_action(
                adapter, query, action, current),
        }
        title, choices, controls = picker_page(state, 0)
        if not eligible:
            title, choices = project_copy("empty"), []
        result = await adapter.send_project_picker(
            chat_id=source.chat_id, token=token, title=title, choices=choices, controls=controls,
            metadata=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
            thread_id=source.thread_id, state=state,
        )
        return None if result.success else project_copy("open_failed")

    async def _handle_project_picker_action(self, adapter, query, action: str, state: dict) -> dict:
        """Revalidate every tap against the current route and registry; only a folder tap commits."""
        profile_home = Path(state["profile_home"])
        if str(self._resolve_profile_home_for_source(state["source"])) != str(profile_home) or (
            self.session_store.peek_session_id(state["session_key"]) != state["expected_session_id"]
        ):
            return {"text": project_copy("stale"), "done": True}
        if action == "cancel":
            return {"text": project_copy("cancelled"), "done": True}
        if action == "back":
            state["view"] = "projects"
            return self._project_picker_view(state, 0, folder=False)
        if action.startswith("page_"):
            try:
                page = int(action[5:])
            except ValueError:
                return {"text": project_copy("invalid"), "done": False}
            return self._project_picker_view(state, page, folder=state["view"] == "folders")
        current = {p["id"]: p for p in eligible_projects(
            await asyncio.to_thread(list_projects_for_profile, profile_home))}
        if action.startswith("select_"):
            return await self._project_picker_select(adapter, query, state, current, action)
        if action.startswith("folder_"):
            try:
                folder = state["folders"][int(action[7:])]
            except (ValueError, IndexError):
                return {"text": project_copy("invalid"), "done": False}
            project = current.get(state["project_id"])
            if project is None or not any(f["path"] == folder["path"] for f in project["folders"]):
                return {"text": project_copy("removed"), "done": True}
            return await self._commit_project_picker_selection(state, project, folder)
        return {"text": project_copy("invalid"), "done": False}

    @staticmethod
    def _project_picker_view(state: dict, page: int, *, folder: bool) -> dict:
        title, choices, controls = picker_page(state, page, folder=folder)
        return {"title": title, "choices": choices, "controls": controls}

    async def _project_picker_select(self, adapter, query, state: dict, current: dict, action: str) -> dict:
        """A project tap. The listed snapshot (indices, order) is NOT replaced: it stays stable."""
        try:
            selected = state["projects"][int(action[7:])]
        except (ValueError, IndexError):
            return {"text": project_copy("invalid"), "done": False}
        project = current.get(selected["id"])
        if project is None:
            return {"text": project_copy("removed"), "done": True}
        state.update(project_id=project["id"], project_name=project["name"], folders=project["folders"])
        if len(project["folders"]) == 1:
            return await self._commit_project_picker_selection(state, project, project["folders"][0])
        state["view"] = "folders"
        return self._project_picker_view(state, 0, folder=True)

    async def _commit_project_picker_selection(self, state: dict, project: dict, folder: dict) -> dict:
        """Validate, then replace the route's session once, under the per-route transition lock.

        Everything that can fail (workspace, backend, DB, CAS) runs BEFORE the running turn is touched;
        the running turn is interrupted and its runtime state cleared only after the replacement is
        durably published. A stale or failed commit leaves the previous turn and route untouched.
        """
        source, route, expected = state["source"], state["session_key"], state["expected_session_id"]
        async with route_transition_lock(self, route), self._async_profile_scope_for_source(source):
            if self.session_store.peek_session_id(route) != expected:
                return {"text": project_copy("stale"), "done": True}
            current = {p["id"]: p for p in eligible_projects(
                await asyncio.to_thread(list_projects_for_profile, Path(state["profile_home"])))}
            folder_now = next((f for f in (current.get(project["id"]) or {}).get("folders", [])
                               if f["path"] == folder["path"]), None)
            if folder_now is None:
                return {"text": project_copy("removed"), "done": True}
            try:
                require_local_backend()
                cwd = await asyncio.to_thread(validate_workspace, folder_now["path"])
            except (ProjectWorkspaceError, OSError, ValueError) as exc:
                return {"text": str(exc), "done": True}
            if self.session_store._db_for_key(route) is None:
                return {"text": project_copy("storage_unavailable"), "done": True}
            from tui_gateway import git_probe
            branch, root = await asyncio.gather(
                asyncio.to_thread(git_probe.branch, cwd), asyncio.to_thread(git_probe.common_repo_root, cwd))
            old_entry = self.session_store.lookup_by_session_key(route)
            try:  # publish first: nothing running is touched unless the replacement is durable
                new_entry = await self.async_session_store.reset_session_for_project(
                    route, source, expected_session_id=expected, cwd=cwd, project_id=project["id"],
                    git_branch=branch, git_repo_root=root)
            except Exception:
                logger.exception("Telegram project session could not be created; previous route kept")
                return {"text": project_copy("create_failed"), "done": True}
            if new_entry is None:
                return {"text": project_copy("stale"), "done": True}
            from gateway.run import _INTERRUPT_REASON_RESET
            # Same busy contract as plain `/new`: cancel the running turn, adapter activity and pending text.
            await self._interrupt_and_clear_session(
                route, source, interrupt_reason=_INTERRUPT_REASON_RESET,
                invalidation_reason="project_session_reset")
            await self._stop_and_clear_route_for_reset(
                route, reason="project_session_reset", old_entry=old_entry)
            await self._finish_project_session_start(source, route, expected, new_entry, old_entry)
            return {"text": project_copy("started", project=project["name"]), "done": True}

    async def _finish_project_session_start(self, source, route, old_sid, new_entry, old_entry) -> None:
        """Post-publish handoff shared with `/new`; each step is best-effort and cannot undo the switch."""
        from gateway.session_automation import move_route_wakes
        try:
            await move_route_wakes(self, route, old_sid)
        except Exception:
            logger.warning("Project session route-wake handoff failed", exc_info=True)
        try:
            if await asyncio.to_thread(self._is_telegram_topic_lane, source):
                await asyncio.to_thread(self._record_telegram_topic_binding, source, new_entry)
        except Exception:
            logger.warning("Project session topic rebind failed", exc_info=True)
        try:
            await self._fire_session_reset_hooks(source, route, old_sid, new_entry.session_id)
        except Exception:
            logger.warning("Project session reset hooks failed", exc_info=True)
        if old_entry is not None:
            try:
                from hermes_cli.lifecycle import invoke_hook
                invoke_hook("on_session_reset", session_id=new_entry.session_id, reason="new_session",
                            platform=source.platform.value, old_session_id=old_sid,
                            new_session_id=new_entry.session_id)
            except Exception:
                logger.debug("Project session on_session_reset hook failed", exc_info=True)
