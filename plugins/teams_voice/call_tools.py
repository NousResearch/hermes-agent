"""CallToolRunner — runs the realtime model's tool calls.

Separates the tool surface (agent consult / background task / look_at_screen /
show_to_caller / call_me_back / post_meeting_minutes) from the realtime handler's
transport + dialogue loop. Holds a reference to the handler for per-call context
(session, vision store/budget, consult, caller, bridge, meeting).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from . import meeting
from .call_session_base import _pending_set
from .outbound import OutboundError, place_call

logger = logging.getLogger(__name__)


class CallToolRunner:
    def __init__(self, handler) -> None:
        self._h = handler

    async def run_tool(self, name: str, args: dict) -> str:
        h = self._h
        try:
            if name == "hermes_agent_consult":
                return await h._consult.ask(str(args.get("query", "")))
            if name == "hermes_agent_task":
                return await self._agent_task(str(args.get("query", "")))
            if name == "look_at_screen":
                return await self._look_at_screen(
                    str(args.get("question", "")), args.get("source"), str(args.get("scope") or "live")
                )
            if name == "show_to_caller":
                return await self._show_to_caller(str(args.get("prompt", "")), args.get("count", 1))
            if name == "call_me_back":
                return await self._call_me_back(str(args.get("message", "")))
            if name == "post_meeting_minutes":
                return await meeting.post_minutes(h._consult, h._meeting, h._thread_id)
        except Exception:  # noqa: BLE001 — a tool fault must not break the call
            logger.error("[teams_voice] tool %s failed", name, exc_info=True)
            return "Sorry, that didn't work."
        return f"Unknown tool: {name}."

    async def _look_at_screen(self, question: str, source: str | None, scope: str = "live") -> str:
        h = self._h
        if not h._vision_budget.try_consume():
            return "I've looked at a lot just now — give me a moment before the next one."
        prompt = question.strip() or "Describe what you see."
        if scope == "history":
            frames = h._vision.history(limit=6)
            if not frames:
                return "I don't have any earlier frames to look back on."
            content: list[dict] = [{"type": "text", "text": prompt}]
            for f in frames:  # timestamped, attributed keyframes
                content.append({"type": "text", "text": f"(earlier, from {f.describe()})"})
                content.append({"type": "image_url", "image_url": {"url": f.data_url()}})
        else:
            want = "camera" if str(source or "").lower() == "camera" else "screenshare"
            frame = h._vision.latest(want) or h._vision.latest()
            if frame is None:
                return "I can't see a shared screen or camera right now."
            content = [
                {"type": "text", "text": f"{prompt} (looking at the {frame.describe()})"},
                {"type": "image_url", "image_url": {"url": frame.data_url()}},
            ]
        return await self._vision_consult(content)

    async def _vision_consult(self, content: list[dict]) -> str:
        try:
            from agent.auxiliary_client import async_call_llm

            resp = await async_call_llm(
                task="vision", messages=[{"role": "user", "content": content}], max_tokens=400
            )
            text = resp.choices[0].message.content if resp and resp.choices else ""
            return (text or "").strip() or "I couldn't quite make that out."
        except Exception:  # noqa: BLE001
            self._h._vision_budget.refund()  # consult failed before the model — give it back
            logger.error("[teams_voice] vision consult failed", exc_info=True)
            return "I had trouble looking at that."

    async def _show_to_caller(self, prompt: str, count: object = 1) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "What would you like me to show?"
        try:
            n = max(1, min(int(count), 3))
        except (TypeError, ValueError):
            n = 1
        try:
            from tools.image_generation_tool import image_generate_tool

            paths: list[str] = []
            for _ in range(n):
                raw = await asyncio.to_thread(
                    lambda: image_generate_tool(prompt=prompt, aspect_ratio="landscape")
                )
                data = json.loads(raw)
                if data.get("success") and data.get("image"):
                    paths.append(data["image"])
            if not paths:
                return "I couldn't create that image."
            # Paced slideshow: 4.5s hold for non-final, 5s for the final image.
            for idx, path in enumerate(paths):
                final = idx == len(paths) - 1
                img_bytes = Path(path).read_bytes()
                mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
                if self._h._session is not None:
                    await self._h._session.send_display_image(
                        base64.b64encode(img_bytes).decode("ascii"),
                        mime,
                        duration_ms=5000 if final else 4500,
                        mode="overlay",
                        caption=prompt[:80],
                    )
                if not final:
                    await asyncio.sleep(4.0)
            return "I'm showing it on screen now." if len(paths) == 1 else f"Showing you {len(paths)} images."
        except Exception:  # noqa: BLE001
            logger.error("[teams_voice] show_to_caller failed", exc_info=True)
            return "I made the image but couldn't display it."

    async def _call_me_back(self, message: str) -> str:
        h = self._h
        message = message.strip()
        caller = h._caller
        if h._bridge is None or caller is None or not caller.aad_id:
            return "I can't call you back — I don't have a number to reach you."
        tenant = caller.tenant_id or h._bridge.tenant_id
        if not tenant:
            return "I can't call you back — missing your tenant."
        try:
            result = await place_call(
                user_object_id=caller.aad_id,
                tenant_id=tenant,
                shared_secret=h._bridge.shared_secret,
                worker_base_url=h._bridge.worker_base_url,
                allow_remote=h._bridge.allow_remote_worker,
            )
        except OutboundError as exc:
            logger.warning("[teams_voice] call_me_back failed: %s", exc)
            return "I couldn't place the call-back just now."
        call_id = result.get("callId")
        if call_id:
            _pending_set(call_id, message or "Here's what you asked for.")
        return "Okay — I'll call you right back with that."

    async def _agent_task(self, query: str) -> str:
        """Run a long job in the background; deliver the result to the Teams chat
        (preferred) or via a voice call-back."""
        h = self._h
        query = query.strip()
        caller = h._caller
        if not query:
            return "What would you like me to work on?"
        # Need either a postable thread (chat delivery) or an AAD id (call-back).
        if h._bridge is None or (not getattr(h, "_thread_id", "") and (caller is None or not caller.aad_id)):
            return await h._consult.ask(query)  # no delivery path → inline
        asyncio.create_task(self._run_background_task(query, caller))
        return "Got it — I'll work on that in the background and send you the result."

    async def _run_background_task(self, query: str, caller) -> None:
        h = self._h
        try:
            result = await h._consult.ask(query, timeout_s=300.0)
        except Exception:  # noqa: BLE001
            logger.error("[teams_voice] background task failed", exc_info=True)
            result = "I couldn't complete that task."
        # Prefer delivering the result to the Teams chat (no call-back needed);
        # fall back to a voice call-back when there's no postable thread.
        thread = getattr(h, "_thread_id", "")
        if thread:
            from .meeting import _deliver_to_teams

            if await _deliver_to_teams(thread, f"✅ {result}"):
                return
        if h._bridge is None or caller is None or not caller.aad_id:
            return
        tenant = caller.tenant_id or h._bridge.tenant_id
        if not tenant:
            return
        try:
            res = await place_call(
                user_object_id=caller.aad_id,
                tenant_id=tenant,
                shared_secret=h._bridge.shared_secret,
                worker_base_url=h._bridge.worker_base_url,
                allow_remote=h._bridge.allow_remote_worker,
            )
        except OutboundError as exc:
            logger.warning("[teams_voice] background callback failed: %s", exc)
            return
        cid = res.get("callId")
        if cid:
            _pending_set(cid, result)
