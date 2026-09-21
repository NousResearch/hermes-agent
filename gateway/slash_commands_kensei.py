"""KENSEI CUSTOM — content-engine + session-utility gateway slash handlers.

Ported from the fork's gateway/slash_commands.py (6,078 lines upstream didn't
have). This mixin keeps the fork-only in-session slash flows alive on the
merged tree:

- /generate-image multi-step wizard (style/aspect/backend via the shared
  in-process content-engine service)
- /localgen (ComfyUI Wan2.2 video)
- /mode + reasoning picker + loop-manager helpers used by the CLI-side
  model switch surface

All other fork slash handlers are covered by upstream's decomposed mixins
(model/session/status/goals).
"""

from __future__ import annotations

import asyncio  # KENSEI: async_session_store awaits in slash handlers
from typing import Union

class GatewayKenseiSlashMixin:
    """Fork-only gateway slash handlers."""

    # ── KENSEI CUSTOM — content-engine slash constants (ported from fork) ──
    # X manager Discord channel: "/x <note>" drops a capture for the thesis
    # incubator (content_engine.x_inbox). Never posts; never drafts.
    X_INBOX_CHANNEL_ID = "1539435276160729148"
    _GI_STYLE_CHOICES: tuple[str, ...] = (
        "data-atlas",
        "mythic-tech-codex",
        "ninth-observatory",
        "chromatic-institute",
        "signal-hud",
        "technical-diorama",
        "typographic-poster-design",
        "vintage-print-atelier",
        "photographic-realism",
        "cosmic-postcard",
        "ink-ember-studio",
        "saga-noir",
        "pixel-art",
    )
    _GI_ASPECT_CHOICES: tuple[str, ...] = ("landscape", "square", "portrait")
    _GI_BACKEND_CHOICES: tuple[str, ...] = ("codex", "local")
    _LG_MODEL_CHOICES: tuple[str, ...] = ("fast-video", "animate")
    # ── END KENSEI CUSTOM ──

    async def _handle_mode_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /mode command — set or query agent execution mode.

        Supports:
          /mode              — show current mode
          /mode status       — show current mode
          /mode auto         — clear mode overlay (back to normal)
          /mode plan         — plan mode
          /mode gods_plan    — UltraPlan mode
          /mode recon        — recon mode

        Mode prompts live in hermes_cli/mode_prompts.py (single source of
        truth).  The mode is persisted via SessionStore.set_agent_mode so
        it survives gateway restarts, and cleared on /new.  The cached
        agent is evicted so the next turn picks up the new ephemeral
        prompt (signature-safe: combined_ephemeral participates in the
        agent-cache signature).
        """
        from hermes_cli.mode_prompts import (
            validate_mode, get_mode_prompt, detect_mode, mode_label,
        )

        source = event.source
        source = await asyncio.to_thread(self._normalize_source_for_session_key, source)
        session_key = self._session_key_for_source(source)

        raw_args = event.get_command_args().strip()
        arg = raw_args.lower() if raw_args else ""

        # /mode or /mode status — report current mode
        if not arg or arg == "status":
            current_mode = await self.async_session_store.get_agent_mode(session_key)
            label = mode_label(current_mode)
            return f"  mode: {label}"

        # Validate — raises ValueError on invalid mode
        try:
            mode = validate_mode(arg)
        except ValueError:
            return "  ❌ Unknown mode: " + raw_args + chr(10) + "  Valid modes: auto, plan, gods_plan, recon"

        # Persist to SessionStore (survives restarts, cleared on /new)
        await self.async_session_store.set_agent_mode(session_key, mode)

        # Evict cached agent so next turn rebuilds with the new
        # ephemeral_prompt (signature-safe via combined_ephemeral).
        self._evict_cached_agent(session_key)

        # If a live agent exists, update its ephemeral_system_prompt now
        # so the change takes effect on the next message without waiting
        # for cache eviction + rebuild.  Guard against the pending sentinel.
        _running = getattr(self, "_running_agents", {}).get(session_key)
        if _running is not None and hasattr(_running, "ephemeral_system_prompt"):
            try:
                _running.ephemeral_system_prompt = get_mode_prompt(mode)
            except Exception:
                pass

        return f"  mode → {mode_label(mode)}"

    async def _get_loop_manager_for_event(self, event: "MessageEvent"):
        """Return a LoopManager bound to the session for this gateway event.

        Returns ``(manager, session_entry)`` or ``(None, None)`` when the
        loops module or session can't be loaded. Mirrors
        ``_get_goal_manager_for_event``.
        """
        try:
            from hermes_cli.loops import LoopManager
        except Exception as exc:
            logger.debug("loop manager unavailable: %s", exc)
            return None, None
        # Warm the SessionDB cache off-loop. A cold cache drops the first
        # /loop write while the reply claims the loop was set (same class
        # as the /goal false-ack fix).
        await self._warm_goals_session_db("loop manager")
        try:
            session_entry = await self.async_session_store.get_or_create_session(event.source)
        except Exception:
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        return LoopManager(session_id=sid), session_entry

    async def _gateway_session_diff(self, cwd: str, stat_only: bool) -> str:
        """Cumulative checkpoint-baseline diff for /diff session (gateway)."""
        from gateway.run import _checkpoint_agent_kwargs, _load_gateway_config
        from tools.checkpoint_manager import CheckpointManager

        cp_kwargs = _checkpoint_agent_kwargs(_load_gateway_config())
        if not cp_kwargs["checkpoints_enabled"]:
            return t("gateway.diff.not_enabled")

        mgr = CheckpointManager(
            enabled=True,
            max_snapshots=cp_kwargs["checkpoint_max_snapshots"],
            max_total_size_mb=cp_kwargs["checkpoint_max_total_size_mb"],
            max_file_size_mb=cp_kwargs["checkpoint_max_file_size_mb"],
        )

        result = await asyncio.to_thread(mgr.session_diff, cwd)
        if not result.get("success"):
            return t("gateway.diff.failed",
                     error=result.get("error", "Could not generate diff"))

        stat = result.get("stat", "")
        diff = result.get("diff", "")
        if result.get("empty") or (not stat and not diff):
            return t("gateway.diff.no_changes")

        out: list[str] = []
        if stat:
            out.append(f"```\n{stat}\n```")
        if not stat_only and diff:
            out.append(self._fenced_truncated_diff(diff))
        return "\n\n".join(out)

    @staticmethod

    def _reasoning_picker_choices(self, current_effort: str) -> list:
        """Build the choice list for the interactive /reasoning picker."""
        from hermes_constants import VALID_REASONING_EFFORTS

        choices = [
            {
                "value": "none",
                "label": t("gateway.reasoning.choice_none"),
                "is_current": current_effort == "none",
            }
        ]
        for level in VALID_REASONING_EFFORTS:
            choices.append(
                {
                    "value": level,
                    "label": level,
                    "is_current": level == current_effort,
                }
            )
        choices.extend(
            [
                {"value": "reset", "label": t("gateway.reasoning.choice_reset"), "is_current": False},
                {"value": "show", "label": t("gateway.reasoning.choice_show"), "is_current": False},
                {"value": "hide", "label": t("gateway.reasoning.choice_hide"), "is_current": False},
            ]
        )
        return choices

    async def _handle_generate_image_command(self, event: "MessageEvent") -> Union[str, "EphemeralReply", None]:
        """Handle /generate-image on the gateway.

        Two paths:

        1. **All fields inline** — when the user supplies every required
           field via ``key=value`` pairs (``prompt=...|style=...|stage-root=...|job-id=...``),
           the handler validates them, shows the final config + exact command,
           and routes through ``_request_slash_confirm`` for a final
           confirmation before invoking the shared in-process content-engine service.

        2. **Multi-step interaction** — when required fields are missing, the
           handler registers a per-session state machine
           (``tools.generate_image_interaction``) and sends the first missing
           field's question via the adapter's ``send_clarify``.  Each user
           reply advances the state machine; when all fields are collected the
           final confirmation is shown.  This works on every platform
           (Telegram buttons, Discord slash command, Slack, text fallback).

        The handler never blocks the event loop — it returns an ack and the
        message intercept in ``_handle_message`` feeds subsequent replies
        into the state machine.  Generation logic is never duplicated: the shared in-process
        content-engine service is the single execution path.
        """
        import json as _json
        import os as _os
        import re as _re
        import sys as _sys
        import uuid as _uuid
        from pathlib import Path as _Path

        # Mirrors content_engine.image_reference_staging._JOB_ID_RE — validated
        # locally before spawning so a bad id is rejected without a subprocess.
        _GI_JOB_ID_RE = _re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")

        raw_args = event.get_command_args().strip()
        fields: dict[str, str] = {}
        if raw_args:
            for pair in raw_args.split("|"):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    fields[k.strip().lower().replace("-", "_")] = v.strip()

        # Keep omitted optional values absent until the interaction has offered
        # them.  The confirmation seam applies codex/landscape defaults only
        # after the user has had a chance to select or decline them.

        # Validate enum fields early when supplied inline.
        # because they have fixed choices (the multi-step flow offers choices
        # for style/aspect/backend, but if the user passed a bad value inline
        # we reject immediately rather than silently overriding).
        backend = fields.get("backend", "codex").strip()
        if backend not in self._GI_BACKEND_CHOICES:
            return EphemeralReply(
                f"Invalid backend '{backend}'. Must be one of: {', '.join(self._GI_BACKEND_CHOICES)}."
            )
        aspect_ratio = fields.get("aspect_ratio", "landscape").strip()
        if aspect_ratio not in self._GI_ASPECT_CHOICES:
            return EphemeralReply(
                f"Invalid aspect-ratio '{aspect_ratio}'. Must be one of: {', '.join(self._GI_ASPECT_CHOICES)}."
            )

        # If the caller supplied the core required values inline, optional
        # backend/references/aspect-ratio use their documented defaults.  When
        # starting a guided flow, ask every field so the user can override
        # those defaults explicitly.
        _core_required = ("prompt", "style", "stage_root", "job_id")
        _all_fields = (
            "prompt", "style", "backend", "references", "stage_root", "job_id", "aspect_ratio", "blend",
        )
        if any(not fields.get(field, "").strip() for field in _core_required):
            missing = [field for field in _all_fields if not fields.get(field, "").strip()]
        else:
            missing = []

        if missing:
            # Multi-step interaction: register the state machine and send the
            # first question.  The message intercept feeds subsequent replies.
            return await self._gi_start_multi_step(event, fields, missing)

        # All fields present — validate job_id before showing confirmation.
        job_id = fields.get("job_id", "").strip()
        if not _GI_JOB_ID_RE.fullmatch(job_id):
            return EphemeralReply(
                f"Invalid job-id '{job_id}'. Must be 1-64 chars, start with a "
                f"letter/digit, and contain only letters, digits, underscores "
                f"or hyphens."
            )

        # Show final config + command, require confirmation.
        return await self._gi_confirm_and_execute(event, fields)

    async def _gi_start_multi_step(
        self, event: "MessageEvent", fields: dict[str, str], missing: list[str],
    ) -> Union[str, "EphemeralReply", None]:
        """Register the multi-step state machine and send the first question."""
        from tools import generate_image_interaction as _gi

        source = event.source
        session_key = self._session_key_for_source(source)
        _gi.register(session_key, fields, missing)
        return await self._gi_ask_next(event, session_key)

    async def _gi_ask_next(
        self, event: "MessageEvent", session_key: str,
    ) -> Union[str, "EphemeralReply", None]:
        """Send the next missing-field question, or trigger final confirmation.

        Reads the pending state for ``session_key``.  If no fields remain
        missing, validates job_id and routes to ``_gi_confirm_and_execute``.
        Otherwise sends a question for the next field (with a numbered list
        for choice-based fields) as a plain text message via the adapter and
        returns.  The message intercept in ``_handle_message`` feeds the
        user's next reply into ``_gi_resolve_step``.

        We deliberately do NOT use the clarify_gateway primitive here: the
        clarify text-intercept in ``_handle_message`` runs *before* our
        generate-image intercept and would consume the reply, preventing the
        state machine from advancing.  Using a plain send + our own intercept
        keeps the flow self-contained and avoids that race.
        """
        from tools import generate_image_interaction as _gi

        state = _gi.get_pending(session_key)
        if state is None:
            return EphemeralReply("Generate-image interaction expired. Please restart with /generate-image.")

        missing = state.get("missing", [])
        fields = dict(state.get("fields", {}))

        if not missing:
            # All fields collected — validate job_id and confirm.
            job_id = fields.get("job_id", "").strip()
            import re as _re
            _GI_JOB_ID_RE = _re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
            if not _GI_JOB_ID_RE.fullmatch(job_id):
                _gi.clear(session_key)
                return EphemeralReply(
                    f"Invalid job-id '{job_id}'. Must be 1-64 chars, start with a "
                    f"letter/digit, and contain only letters, digits, underscores "
                    f"or hyphens. Please restart with /generate-image."
                )
            return await self._gi_confirm_and_execute(event, fields)

        field = missing[0]
        question, choices = self._gi_question_for_field(field, fields)

        # Build a numbered list for choice-based fields so users on text-only
        # platforms (Telegram, Slack) get a selectable menu.
        if choices:
            lines = [f"❓ {question}", ""]
            for i, choice in enumerate(choices, start=1):
                lines.append(f"  {i}. {choice}")
            lines.append("")
            lines.append("Reply with the number, the option text, or your own value.")
            message_text = "\n".join(lines)
        else:
            message_text = f"❓ {question}"

        # Send via the adapter as a plain message.  The reply is caught by
        # our generate-image intercept in _handle_message.
        adapter = self._adapter_for_source(event.source)
        metadata = self._thread_metadata_for_source(
            event.source, self._reply_anchor_for_event(event),
        )
        if adapter is not None:
            try:
                await adapter.send(
                    chat_id=str(event.source.chat_id),
                    content=message_text,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.debug("generate-image question send failed: %s", exc)
                return message_text
            return None

        # No adapter — return the text so the caller can display it.
        return message_text

    def _gi_question_for_field(
        self, field: str, fields: dict[str, str],
    ) -> tuple[str, Optional[list[str]]]:
        """Return (question, choices) for the next missing field."""
        if field == "prompt":
            return ("What prompt should the image use?", None)
        if field == "style":
            return (
                "Which style?",
                list(self._GI_STYLE_CHOICES),
            )
        if field == "backend":
            return ("Which backend? Press Enter for codex.", list(self._GI_BACKEND_CHOICES))
        if field == "references":
            return ("Reference URLs, comma-separated (or press Enter for none)?", None)
        if field == "stage_root":
            return (
                "What private stage-root path should the job use? (absolute path)",
                None,
            )
        if field == "job_id":
            return (
                "What safe job ID should this image use? "
                "(letters, digits, underscores, hyphens; e.g. gen-20260101-abc)",
                None,
            )
        if field == "aspect_ratio":
            return (f"Which aspect ratio? Press Enter for landscape.", list(self._GI_ASPECT_CHOICES))
        if field == "blend":
            return (
                "Blend two or more registry styles? e.g. `steampunk+synthwave` (or press Enter for none).",
                None,
            )
        # Unknown field — open-ended.
        return (f"Value for {field}?", None)

    async def _gi_resolve_step(
        self, event: "MessageEvent",
    ) -> Optional[str]:
        """Feed a user reply into a pending generate-image interaction.

        Called from the message intercept when a pending interaction exists.
        Returns a reply string if the step was resolved (possibly the next
        question or the final confirmation), or None to fall through.
        """
        from tools import generate_image_interaction as _gi

        source = event.source
        session_key = self._session_key_for_source(source)
        state = _gi.get_pending(session_key)
        if state is None:
            return None

        raw_reply = (event.text or "").strip()
        # Slash commands bypass — the user wanted to issue a command, not
        # answer the prompt.
        if not raw_reply or raw_reply.startswith("/"):
            return None

        missing = state.get("missing", [])
        if not missing:
            return None

        field = missing[0]

        # Choice fields accept a numbered or exact-label selection. Style may
        # also be a custom typed value; empty backend/aspect replies take their
        # documented defaults.
        if field == "style":
            value = self._gi_coerce_choice(raw_reply, self._GI_STYLE_CHOICES) or raw_reply
        elif field == "backend":
            value = self._gi_coerce_choice(raw_reply, self._GI_BACKEND_CHOICES) or ("codex" if not raw_reply else "")
            if not value:
                return "Invalid backend. Reply with codex or local."
        elif field == "aspect_ratio":
            value = self._gi_coerce_choice(raw_reply, self._GI_ASPECT_CHOICES) or ("landscape" if not raw_reply else "")
            if not value:
                return "Invalid aspect ratio. Reply with landscape, square, or portrait."
        elif field == "references":
            # Chat platforms cannot submit a meaningful empty reply; accept a
            # natural explicit opt-out and preserve an empty reference list.
            value = "" if raw_reply.casefold() in {"none", "no", "skip", "-"} else raw_reply
        elif field == "blend":
            # Accept comma- or plus-separated registry slugs; empty/none → no blend.
            if raw_reply.casefold() in {"none", "no", "skip", "-", ""}:
                value = ""
            else:
                value = raw_reply
        else:
            value = raw_reply
        updated = _gi.advance(session_key, value)
        if updated is None:
            return None

        # Acknowledge the step, then ask the next question or confirm.
        still_missing = updated.get("missing", [])
        if still_missing:
            return await self._gi_ask_next(event, session_key)

        # All fields collected — validate job_id before confirming, mirroring
        # the inline and _gi_ask_next paths.  A bad job_id must be rejected
        # here too, not only when it was supplied inline.
        fields = dict(updated.get("fields", {}))
        import re as _re
        _resolve_job_id_re = _re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
        job_id_val = fields.get("job_id", "").strip()
        if not _resolve_job_id_re.fullmatch(job_id_val):
            _gi.clear(session_key)
            return EphemeralReply(
                f"Invalid job-id '{job_id_val}'. Must be 1-64 chars, start with a "
                f"letter/digit, and contain only letters, digits, underscores "
                f"or hyphens. Please restart with /generate-image."
            )
        # Clear the interaction state before confirming so a /cancel in the
        # confirm prompt doesn't leave a stale pending interaction.
        _gi.clear(session_key)
        return await self._gi_confirm_and_execute(event, fields)

    @staticmethod

    def _gi_coerce_choice(text: str, choices: tuple[str, ...]) -> Optional[str]:
        """Map a typed reply to a choice label, or None if unrecognised."""
        t = text.strip()
        if not t:
            return None
        # Numeric selection (1-based).
        if t.isdigit():
            idx = int(t)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            return None
        # Exact label match (case-insensitive).
        for choice in choices:
            if t.casefold() == choice.casefold():
                return choice
        return None

    async def _gi_confirm_and_execute(
        self, event: "MessageEvent", fields: dict[str, str],
    ) -> Union[str, "EphemeralReply", None]:
        """Show the final config, require confirmation, then execute.

        Routes through ``_request_slash_confirm`` so the confirmation uses
        the platform's button UI (Approve Once / Cancel) where supported,
        with a text fallback.  The ``_on_confirm`` handler invokes the
        shared in-process runner (``tools.generate_image_runner``) on
        approval — never shells out to ``content_engine.py``.
        """
        prompt = fields.get("prompt", "").strip()
        style = fields.get("style", "").strip()
        backend = fields.get("backend", "codex").strip() or "codex"
        stage_root = fields.get("stage_root", "").strip()
        job_id = fields.get("job_id", "").strip()
        aspect_ratio = fields.get("aspect_ratio", "landscape").strip() or "landscape"
        refs_raw = fields.get("references", "").strip()
        references: list[str] = []
        if refs_raw:
            references = [r.strip() for r in refs_raw.split(",") if r.strip()]
        blend_raw = fields.get("blend", "").strip()
        blend: list[str] = []
        if blend_raw:
            blend = [s.strip() for s in blend_raw.replace(",", "+").split("+") if s.strip()]

        # Final configuration summary + exact canonical command shown to the
        # user before confirmation.  The exact command lets the user verify
        # what will actually run (safe quoting, repeated --reference flags).
        from tools.generate_image_runner import render_generate_image_command

        exact_cmd = render_generate_image_command(
            prompt=prompt,
            style=style,
            backend=backend,
            references=references,
            blend=blend,
            stage_root=stage_root,
            job_id=job_id,
            aspect_ratio=aspect_ratio,
        )
        config_summary = (
            f"**generate-image configuration**\n"
            f"• prompt: `{prompt}`\n"
            f"• style: `{style}`\n"
            f"• backend: `{backend}`\n"
            f"• references: {', '.join(references) if references else '(none)'}\n"
            f"• blend: {', '.join(blend) if blend else '(none)'}\n"
            f"• stage-root: `{stage_root}`\n"
            f"• job-id: `{job_id}`\n"
            f"• aspect-ratio: `{aspect_ratio}`\n"
            f"\n"
            f"**exact command**\n"
            f"`{exact_cmd}`"
        )

        async def _on_confirm(choice: str) -> Optional[str]:
            if choice == "cancel":
                return "🟡 generate-image cancelled. No image generated."
            # once / always → execute via the shared in-process runner.
            return await self._gi_execute(
                prompt=prompt,
                style=style,
                backend=backend,
                references=references,
                blend=blend,
                stage_root=stage_root,
                job_id=job_id,
                aspect_ratio=aspect_ratio,
            )

        return await self._request_slash_confirm(
            event=event,
            command="generate-image",
            title="/generate-image — confirm",
            message=config_summary,
            handler=_on_confirm,
        )

    async def _gi_execute(
        self,
        *,
        prompt: str,
        style: str,
        backend: str,
        references: list[str],
        blend: list[str],
        stage_root: str,
        job_id: str,
        aspect_ratio: str,
    ) -> str:
        """Run the shared in-process image runner and return a summary string.

        The content-engine service is synchronous, so the call runs in a
        thread executor to avoid blocking the gateway event loop.  No
        subprocess is spawned.
        """
        import asyncio as _asyncio
        import functools as _functools

        from tools.generate_image_runner import run_generate_image

        loop = _asyncio.get_running_loop()
        try:
            payload = await loop.run_in_executor(
                None,
                _functools.partial(
                    run_generate_image,
                    prompt=prompt,
                    style=style,
                    backend=backend,
                    references=references,
                    blend=blend,
                    stage_root=stage_root,
                    job_id=job_id,
                    aspect_ratio=aspect_ratio,
                ),
            )
        except Exception as exc:
            return f"Image generation failed: {exc}"

        provider = (payload.get("backend") or {}).get("provider", "?")
        model = (payload.get("backend") or {}).get("model", "?")
        output_path = payload.get("output_path", "")
        sha = payload.get("sha256", "")

        return (
            f"✅ Image generated (job {payload.get('job_id', job_id)})\n"
            f"backend: {provider}/{model}\n"
            f"output: {output_path}\n"
            f"sha256: {sha}"
        )

    # /localgen — gateway-side local video generation via ComfyUI (Wan2.2)
    # ------------------------------------------------------------------

    # Offered model ids, kept inline so the mixin never imports the backend at
    # module load time. Mirrors the CLI handler's _MODEL_CHOICES.
    _LG_MODEL_CHOICES: tuple[str, ...] = ("fast-video", "animate")

    async def _handle_localgen_command(self, event: "MessageEvent") -> Union[str, "EphemeralReply", None]:
        """Handle /localgen on the gateway.

        Single-shot flow (no multi-step interaction, unlike /generate-image):
        parse ``key=value|key=value`` inline args; if the core required values
        are present, validate and route to a confirmation prompt; otherwise
        return a plain-text usage hint so the user can supply them. We do not
        open the guided state-machine because the surface is tiny and the user
        is a ComfyUI novice.

        Never blocks the event loop: the runner is synchronous, so it runs in a
        thread executor. The backend (``tools.localgen_runner``) shells out to
        the Sirvir fleet's ``localgen.py`` and parses its JSON — no ComfyUI
        imports in the gateway process.
        """
        import re as _re

        raw_args = event.get_command_args().strip()
        fields: dict[str, str] = {}
        if raw_args:
            for pair in raw_args.split("|"):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    fields[k.strip().lower().replace("-", "_")] = v.strip()

        model = fields.get("model", "fast-video").strip()
        if model not in self._LG_MODEL_CHOICES:
            return EphemeralReply(
                "Unknown model. Use model=fast-video (text→video) or "
                "model=animate (photo→video). Example:\n"
                "/localgen model=fast-video|prompt=a red car at sunset"
            )

        prompt = fields.get("prompt", "").strip()
        if not prompt:
            return EphemeralReply(
                "A prompt is required. Example:\n"
                "/localgen model=fast-video|prompt=a red car at sunset\n"
                "For animate, also add image=/path/to/photo.png"
            )

        image = fields.get("image", "").strip() or None
        if model == "animate" and not image:
            return EphemeralReply(
                "model=animate needs image=<path or URL>. Example:\n"
                "/localgen model=animate|prompt=the car drives forward|image=/path/to/photo.png"
            )

        seed_raw = fields.get("seed", "").strip()
        seed = int(seed_raw) if seed_raw.isdigit() else None
        length_raw = fields.get("length", "").strip()
        length = int(length_raw) if length_raw.isdigit() else None
        output_dir = fields.get("output_dir", "").strip() or None

        config_summary = (
            f"**localgen configuration**\n"
            f"• model: `{model}`\n"
            f"• prompt: `{prompt}`\n"
            f"• image: {image or '(none)'}\n"
            f"• seed: {seed if seed is not None else '(random)'}\n"
            f"• length: {length if length is not None else 49} frames\n"
        )

        async def _on_confirm(choice: str) -> Optional[str]:
            if choice == "cancel":
                return "🟡 localgen cancelled. No video generated."
            return await self._lg_execute(
                model=model, prompt=prompt, image=image,
                seed=seed, length=length, output_dir=output_dir,
            )

        return await self._request_slash_confirm(
            event=event,
            command="localgen",
            title="/localgen — confirm",
            message=config_summary,
            handler=_on_confirm,
        )

    async def _lg_execute(
        self,
        *,
        model: str,
        prompt: str,
        image: Optional[str],
        seed: Optional[int],
        length: Optional[int],
        output_dir: Optional[str],
    ) -> str:
        """Run the localgen runner in an executor and return a summary string.

        Mirrors ``_gi_execute``: synchronous backend, thread executor, JSON
        parsed into a plain-text result. Never shells a subprocess directly
        here — ``tools.localgen_runner`` owns the subprocess.
        """
        import asyncio as _asyncio
        import functools as _functools

        from tools.localgen_runner import (
            LOCALGEN_ERROR_MESSAGES,
            LocalGenError,
            run_localgen,
        )

        loop = _asyncio.get_running_loop()
        try:
            payload = await loop.run_in_executor(
                None,
                _functools.partial(
                    run_localgen,
                    model=model,
                    prompt=prompt,
                    image=image,
                    seed=seed,
                    length=length,
                    output_dir=output_dir,
                ),
            )
        except LocalGenError as exc:
            msg = LOCALGEN_ERROR_MESSAGES.get(exc.error_code, str(exc.detail))
            return f"Video generation failed: {msg}"

        if not isinstance(payload, dict) or payload.get("status") != "success":
            code = (payload or {}).get("error_code", "unknown")
            detail = LOCALGEN_ERROR_MESSAGES.get(
                code, (payload or {}).get("error", "Generation failed."))
            return f"Video generation failed: {detail}"

        outputs = payload.get("outputs", [])
        files = "\n".join(f"• {o.get('file', '')}" for o in outputs) or "• (no files)"
        return (
            f"✅ Video generated (model {payload.get('model', model)})\n"
            f"elapsed: {payload.get('elapsed_seconds', '?')}s\n"
            f"outputs:\n{files}"
        )


    # ── KENSEI CUSTOM — channel trigger helpers (ported) ──

    def _is_x_inbox_message(self, event: "MessageEvent") -> bool:
        """True when this message is a /x capture in the X manager channel."""
        text = (event.text or "").strip()
        if not text.lower().startswith("/x"):
            return False
        if len(text) > 2 and not text[2].isspace():
            return False
        source_chat = str(getattr(event.source, "chat_id", "") or "")
        return source_chat == self.X_INBOX_CHANNEL_ID

    async def _handle_x_inbox_capture(self, event: "MessageEvent") -> str:
        """Store the /x capture and reply with a lightweight ack."""
        text = (event.text or "").strip()
        note = text[2:].strip() if len(text) > 2 else ""
        if not note:
            return ""
        try:
            import x_inbox  # type: ignore[import-not-found]
            cap_id = x_inbox.add_capture(
                note,
                source="user",
                channel=str(getattr(event.source, "chat_id", "") or ""),
                author_id=str(getattr(event.source, "user_id", "") or ""),
            )
        except Exception as exc:
            self._logger.warning("x-inbox capture failed: %s", exc)
            return ""
        try:
            adapter = self._adapter_for_source(event.source)
            if adapter is not None:
                await adapter.send(
                    self.X_INBOX_CHANNEL_ID,
                    f"Captured. `{cap_id}`",
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("x-inbox ack failed: %s", exc)
        return ""

    def _image_lab_channel_id(self) -> Optional[str]:
        import os as _os
        return (_os.getenv("IMAGE_LAB_CHANNEL_ID") or "").strip() or None

    async def _is_image_lab_message(self, event: "MessageEvent") -> bool:
        """True when this message is a plain-text image prompt in the lab channel.

        Rules: the message starts with ``img`` (case-insensitive) followed by
        whitespace or is exactly ``img``, and the channel matches
        IMAGE_LAB_CHANNEL_ID when configured.  Without a configured channel,
        the trigger only fires when the adapter reports free-response for the
        source channel (so it never hijacks normal @-mention chat).
        """
        text = (event.text or "").strip()
        if not text.lower().startswith("img"):
            return False
        if len(text) > 3 and not text[3].isspace():
            return False
        # Channel-scope guard.
        lab_id = self._image_lab_channel_id()
        source_chat = str(getattr(event.source, "chat_id", "") or "")
        if lab_id:
            return source_chat == lab_id
        # No explicit channel: only fire in free-response channels.
        try:
            adapter = self._adapter_for_source(event.source)
            if adapter is None:
                return False
            free_channels = adapter._discord_free_response_channels()
            return source_chat in [str(c) for c in free_channels]
        except Exception:
            return False

    def _image_lab_event(self, event: "MessageEvent", prompt: str) -> "MessageEvent":
        """Return a synthetic event whose command args feed /generate-image.

        The synthetic args carry the prompt plus safe lab defaults so the
        handler's inline path runs (confirm → execute → post-back).  The
        source is the original message's channel, so the image is posted back
        to the same channel.
        """
        import time as _time
        import uuid as _uuid
        lab_id = self._image_lab_channel_id() or str(getattr(event.source, "chat_id", ""))
        job_id = f"lab-{_uuid.uuid4().hex[:10]}"
        stage_root = f"/tmp/gi-lab-{lab_id.replace('#', '')}"
        args = (
            f"prompt={prompt}|"
            f"style=mythic-tech-codex|"
            f"backend=codex|"
            f"stage_root={stage_root}|"
            f"job_id={job_id}|"
            f"aspect_ratio=landscape"
        )
        # Copy the event and override get_command_args.  The event is a
        # lightweight dataclass; build a shallow copy via object.__new__ +
        # __dict__ copy to avoid deep-copy costs on a hot path.
        clone = object.__new__(type(event))
        clone.__dict__ = dict(getattr(event, "__dict__", {}))
        setattr(clone, "_gi_lab_args", args)
        original = getattr(event, "get_command_args", None)
        if original is not None:
            import types as _types
            def _args_override(self_obj):  # noqa: ANN001
                return getattr(self_obj, "_gi_lab_args", "")
            clone.get_command_args = _types.MethodType(_args_override, clone)
        else:
            # Fallback: attach a method on the instance.
            setattr(clone, "_gi_lab_args", args)
            clone.get_command_args = lambda: args  # type: ignore[method-assign]
        return clone
    # ── END KENSEI CUSTOM ──
