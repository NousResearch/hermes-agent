"""Petdex mascot (base-CLI pet pane) methods for ``HermesCLI``.

Extracted from ``cli.py`` as part of the god-file decomposition campaign
(epic #78647, target #55136). This mixin holds the Petdex mascot cluster:
config resolution, reaction flashes, animation thread, and the half-block
fragment/window-height render helpers.

Behavior-neutral: every method is lifted verbatim from ``HermesCLI``.
``self.*`` calls resolve unchanged via the MRO. Import discipline mirrors
the other ``cli_*_mixin`` modules: neutral stdlib dependencies (``time``,
``threading``) are imported at module top level; everything else
(``agent.pet.*``, ``hermes_cli.config.load_config``) is imported lazily
inside each method exactly as in ``cli.py``. This module never imports
``cli`` at any time -> no import cycle.
"""

from __future__ import annotations

import threading
import time


class CLIPetMixin:
    """Petdex mascot (base-CLI pet pane) methods for ``HermesCLI``."""

    # ── Petdex mascot (base-CLI pet pane) ───────────────────────────────
    #
    # Parity with the TUI: a half-block sprite rendered as a prompt_toolkit
    # window above the prompt, reacting to agent state and animated by a timer
    # that calls ``app.invalidate()``. Half-blocks only — the crisp Kitty image
    # protocol can't coexist with prompt_toolkit's patch_stdout output layer
    # (raw image escapes get swallowed/mangled), so we use truecolor styled
    # text, which prompt_toolkit renders natively in any 24-bit terminal.

    _PET_FRAME_INTERVAL = 0.16
    _PET_CFG_INTERVAL = 2.5

    def _pet_resolve_config(self) -> None:
        """(Re)resolve the active pet from config — picks up live enable/disable/

        switch made via ``/pet`` or ``hermes pets`` without a restart, mirroring
        the TUI's steady poll. Cheap and fail-open: any problem disables the pet.
        """
        try:
            from agent.pet import constants, store
            from agent.pet.render import PetRenderer
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}

            enabled = bool(pet_cfg.get("enabled"))
            slug = str(pet_cfg.get("slug", "") or "")
            scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
            cols = constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))

            if not enabled:
                with self._pet_lock:
                    self._pet_enabled = False
                    self._pet_renderer = None
                    self._pet_frames_cache.clear()
                return

            pet = store.resolve_active_pet(slug)
            if pet is None or not pet.exists:
                with self._pet_lock:
                    self._pet_enabled = False
                    self._pet_renderer = None
                    self._pet_frames_cache.clear()
                return

            with self._pet_lock:
                # Rebuild only when the resolved pet or geometry changes.
                if (
                    self._pet_renderer is None
                    or self._pet_slug != pet.slug
                    or self._pet_cols != cols
                    or self._pet_scale != scale
                ):
                    self._pet_renderer = PetRenderer(
                        str(pet.spritesheet), mode="unicode", scale=scale, unicode_cols=cols
                    )
                    self._pet_slug = pet.slug
                    self._pet_cols = cols
                    self._pet_scale = scale
                    self._pet_frames_cache.clear()
                    self._pet_frame_idx = 0
                self._pet_enabled = True
        except Exception:
            with self._pet_lock:
                self._pet_enabled = False
                self._pet_renderer = None

    def _pet_flash(self, state: str, secs: float = 1.6) -> None:
        """Briefly force a transient reaction (wave/jump/failed) before resting."""
        self._pet_event = state
        self._pet_event_until = time.monotonic() + secs

    def _on_reaction(self, kind: str) -> None:
        """User affection (ily / <3 / good bot), core-detected — the pet's share
        of the vibe signal that plays hearts on the TUI/desktop. Flash a celebrate."""
        if kind == "vibe":
            self._pet_flash("jump")

    def _pet_react_turn_end(self) -> None:
        """Flash the end-of-turn beat: failed on error, jump on a finished plan, else wave."""
        if not self._pet_enabled:
            return
        from agent.pet.state import todos_all_done

        if self._pet_turn_error:
            self._pet_flash("failed")
            return
        try:
            store = getattr(self.agent, "_todo_store", None)
            done = todos_all_done(store.read()) if store else False
        except Exception:
            done = False
        self._pet_flash("jump" if done else "wave")

    def _derive_pet_state(self) -> str:
        """Map current CLI activity to a pet animation state.

        A transient reaction beat (wave/jump/failed) wins while it's live;
        otherwise the steady state comes from the shared
        :func:`agent.pet.state.derive_pet_state` so the CLI can't drift from the
        TUI/desktop priority order.
        """
        if self._pet_event and time.monotonic() < self._pet_event_until:
            return self._pet_event
        self._pet_event = ""
        from agent.pet.state import derive_pet_state

        # A live blocking modal (approval / clarify / sudo / secret / slash
        # confirm) means the agent is paused on the user → the `waiting` pose,
        # which outranks the in-flight signals in derive_pet_state.
        awaiting_input = bool(
            self._approval_state
            or self._clarify_state
            or self._sudo_state
            or self._secret_state
            or getattr(self, "_slash_confirm_state", None)
        )

        return derive_pet_state(
            awaiting_input=awaiting_input,
            busy=getattr(self, "_agent_running", False),
            reasoning=self._pet_reasoning,
        ).value

    def _pet_frames_for(self, state: str) -> list:
        """Return (and cache) the half-block grids for one state."""
        cached = self._pet_frames_cache.get(state)
        if cached is not None:
            return cached
        renderer = self._pet_renderer
        if renderer is None:
            return []
        try:
            count = renderer.frame_count(state) or 1
            grids = [renderer.cells(state, i, cols=self._pet_cols) for i in range(count)]
        except Exception:
            grids = []
        self._pet_frames_cache[state] = grids
        return grids

    def _pet_fragments(self):
        """Return prompt_toolkit FormattedText for the current pet frame, or []."""
        with self._pet_lock:
            if not self._pet_enabled or self._pet_renderer is None:
                return []
            state = self._derive_pet_state()
            grids = self._pet_frames_for(state)
            if not grids:
                return []
            grid = grids[self._pet_frame_idx % len(grids)]

        frags = []
        for y, row in enumerate(grid):
            if y:
                frags.append(("", "\n"))
            for top, bottom in row:
                tr, tg, tb, ta = top
                br, bg, bb, ba = bottom
                top_op = ta >= 32
                bot_op = ba >= 32
                if not top_op and not bot_op:
                    frags.append(("", " "))
                elif top_op and bot_op:
                    frags.append((f"fg:#{tr:02x}{tg:02x}{tb:02x} bg:#{br:02x}{bg:02x}{bb:02x}", "▀"))
                elif top_op:
                    # Upper half only — leave the lower half the terminal's bg
                    # instead of painting it black (cleaner on light themes).
                    frags.append((f"fg:#{tr:02x}{tg:02x}{tb:02x}", "▀"))
                else:
                    frags.append((f"fg:#{br:02x}{bg:02x}{bb:02x}", "▄"))
        return frags

    def _pet_widget_height(self) -> int:
        """Visible rows for the pet window — 0 collapses it when no pet shows."""
        with self._pet_lock:
            if not self._pet_enabled or self._pet_renderer is None:
                return 0
            grids = self._pet_frames_for(self._derive_pet_state())
            if not grids or not grids[0]:
                return 0
            return len(grids[0])

    def _pet_anim_loop(self) -> None:
        """Advance the frame + invalidate on a timer while a pet is enabled."""
        while self._pet_anim_running:
            time.sleep(self._PET_FRAME_INTERVAL)
            now = time.monotonic()
            if now - self._pet_cfg_checked >= self._PET_CFG_INTERVAL:
                self._pet_cfg_checked = now
                self._pet_resolve_config()
            if not self._pet_enabled:
                continue
            with self._pet_lock:
                self._pet_frame_idx += 1
            app = getattr(self, "_app", None)
            if app is not None:
                try:
                    app.invalidate()
                except Exception:
                    pass

    def _pet_start_anim(self) -> None:
        if self._pet_anim_running:
            return
        self._pet_resolve_config()
        self._pet_anim_running = True
        self._pet_anim_thread = threading.Thread(target=self._pet_anim_loop, daemon=True)
        self._pet_anim_thread.start()

    def _pet_stop_anim(self) -> None:
        self._pet_anim_running = False
        thread = self._pet_anim_thread
        if thread is not None:
            thread.join(timeout=0.3)
        self._pet_anim_thread = None
