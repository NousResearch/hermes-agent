"""Gateway-level orchestration for opportunistic image analysis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from gateway.auto_vision_runtime_service import auto_vision_degraded_note
from gateway.platforms.base import MessageEvent

_DEFAULT_INLINE_WAIT_SECONDS = 0.75
_DEFAULT_GROUP_INLINE_WAIT_SECONDS = 0.25
_DEFAULT_IMAGE_ONLY_INLINE_WAIT_SECONDS = 8.0


@dataclass
class VisionTurnOutcome:
    enriched_text: str
    direct_reply: Optional[str] = None


class VisionOrchestrator:
    def __init__(self, *, config_loader: Callable[[], Dict[str, Any]]):
        self._config_loader = config_loader

    async def prepare_turn(
        self,
        *,
        event: MessageEvent,
        user_text: str,
        analyze_image: Callable[..., Awaitable[Dict[str, Any]]],
    ) -> VisionTurnOutcome:
        images = [
            attachment
            for attachment in event.ensure_attachments()
            if attachment.kind == "image" and not bool(attachment.is_animated)
        ]
        if not images:
            return VisionTurnOutcome(enriched_text=user_text)

        has_user_text = bool(str(user_text or "").strip())
        timeout = self._inline_wait_seconds(
            chat_type=str(getattr(event.source, "chat_type", "") or "").strip().lower(),
            has_user_text=has_user_text,
        )

        prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )
        tasks: list[tuple[str, asyncio.Task[Dict[str, Any]]]] = [
            (
                str(attachment.analysis_ref or attachment.remote_url or attachment.local_path),
                asyncio.create_task(
                    analyze_image(
                        image_ref=str(
                            attachment.analysis_ref or attachment.remote_url or attachment.local_path
                        ),
                        user_prompt=prompt,
                        model=None,
                    )
                ),
            )
            for attachment in images
        ]

        completed_tasks: set[asyncio.Task[Dict[str, Any]]] = set()
        if tasks and timeout > 0:
            done, pending = await asyncio.wait(
                [task for _, task in tasks],
                timeout=timeout,
            )
            completed_tasks = set(done)
            for task in pending:
                task.cancel()
        elif tasks:
            for _, task in tasks:
                task.cancel()

        enriched_parts: list[str] = []
        for image_ref, task in tasks:
            task_result: Dict[str, Any] | None = None
            if task in completed_tasks or task.done():
                try:
                    task_result = task.result()
                except Exception:
                    task_result = None

            if task_result and bool(task_result.get("success")):
                description = str(task_result.get("analysis") or "").strip()
                if description:
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]"
                    )
                    continue

            if task.done():
                enriched_parts.append(auto_vision_degraded_note(image_ref, pending=False))
            else:
                enriched_parts.append(auto_vision_degraded_note(image_ref, pending=True))

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return VisionTurnOutcome(enriched_text=f"{prefix}\n\n{user_text}")
            return VisionTurnOutcome(enriched_text=prefix)

        return VisionTurnOutcome(enriched_text=user_text)

    def _inline_wait_seconds(self, *, chat_type: str, has_user_text: bool) -> float:
        cfg = (((self._config_loader() or {}).get("auxiliary") or {}).get("vision") or {})
        wait_seconds = float(cfg.get("auto_inline_wait", _DEFAULT_INLINE_WAIT_SECONDS))
        group_wait = float(_DEFAULT_GROUP_INLINE_WAIT_SECONDS)
        image_only_wait = float(
            cfg.get("image_only_inline_wait", _DEFAULT_IMAGE_ONLY_INLINE_WAIT_SECONDS)
        )
        if not has_user_text:
            return max(0.0, image_only_wait)
        if chat_type and chat_type != "dm":
            return max(0.0, min(wait_seconds, group_wait))
        return max(0.0, wait_seconds)
