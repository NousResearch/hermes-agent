"""Usage, insights and context-display handlers for the interactive CLI (god-file decomposition).

This module hosts the usage/insights/context-display methods lifted out of
``cli.py``'s ``HermesCLI`` class. ``HermesCLI`` inherits ``CLIUsageMixin`` so
every ``self.<handler>`` call resolves unchanged via the MRO — this cluster is
read-only (no ``self.*`` writes), so the move is behavior-neutral.

Import discipline mirrors ``hermes_cli.cli_billing_mixin``:
  * Neutral, non-cyclic dependencies are imported at module top level below.
  * cli.py-internal symbols (``format_duration_compact``) are imported LAZILY
    inside the method via ``from cli import ...``. The mixin never imports
    ``cli`` at module load time, avoiding the cycle created when ``cli.py``
    imports this mixin.
"""

from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime


class CLIUsageMixin:
    """Mixin holding interactive-CLI usage, insights and context-display handlers."""

    def _handle_usage_command(self, cmd_original: str):
        """Dispatch `/usage [reset [--force]]`.

        Bare `/usage` keeps the classic display. `/usage reset` redeems one
        banked Codex rate-limit reset credit (guarded: refuses when limits
        aren't exhausted unless --force).
        """
        parts = cmd_original.split()
        args = [p.lower() for p in parts[1:]]
        if args and args[0] == "reset":
            self._usage_reset(force="--force" in args[1:])
            return
        if args:
            print(f"  Unknown /usage subcommand: {' '.join(parts[1:])}. Try /usage or /usage reset [--force].")
            return
        self._show_usage()

    def _usage_reset(self, force: bool = False):
        """`/usage reset [--force]` — redeem one banked Codex reset credit."""
        provider = (
            (getattr(self.agent, "provider", None) if self.agent else None)
            or getattr(self, "provider", None)
        )
        normalized = str(provider or "").strip().lower()
        if normalized != "openai-codex":
            print("  Banked usage resets are only available on the openai-codex provider.")
            print("  Switch with `/model` or `hermes auth` first.")
            return
        base_url = (getattr(self.agent, "base_url", None) if self.agent else None) or getattr(self, "base_url", None)
        api_key = (getattr(self.agent, "api_key", None) if self.agent else None) or getattr(self, "api_key", None)

        from agent.account_usage import redeem_codex_reset_credit

        print("  ⏳ Checking banked reset credits...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            try:
                result = _pool.submit(
                    redeem_codex_reset_credit,
                    base_url=base_url,
                    api_key=api_key,
                    force=force,
                ).result(timeout=45.0)
            except concurrent.futures.TimeoutError:
                print("  ❌ Timed out talking to the Codex backend — try again shortly.")
                return
        print(f"  {result.message}")

    def _show_context_breakdown(self, cmd_original: str = ""):
        """`/context [all]` — visual context-window usage breakdown.

        Renders a 5×20 glyph block grid (each cell ≈ 1% of the model context
        window) plus an estimated per-category table: system prompt, tool
        definitions, rules, skills index, MCP, subagents, memory, and the
        conversation itself — versus free space. `/context all` appends the
        expanded per-skill and per-toolset cost listings.

        Read-only: same chars/4 estimation engine as the desktop context
        popover (agent.context_breakdown) — no provider calls, no prompt-cache
        impact.
        """
        if not self.agent:
            print("  (._.) No active agent -- send a message first.")
            return

        args = cmd_original.split(maxsplit=1)[1].strip().lower() if " " in cmd_original else ""
        expanded = args in {"all", "full", "details"}

        from agent.context_breakdown import (
            compute_context_details,
            compute_session_context_breakdown,
            render_context_breakdown_lines,
        )

        try:
            payload = compute_session_context_breakdown(
                self.agent, self.conversation_history
            )
        except Exception as e:
            print(f"  (._.) Could not compute context breakdown: {e}")
            return

        details = None
        if expanded:
            try:
                details = compute_context_details(self.agent)
            except Exception:
                details = {"skills": [], "toolsets": []}

        model = payload.get("model") or self.model
        print()
        print(f"  🧠 Context Usage — {model}")
        print()
        for line in render_context_breakdown_lines(payload, details=details, grid=True):
            print(f"  {line}")
        print()

    def _show_usage(self):
        """Rate limits + session token usage (when a live agent exists) + Nous credits.

        The Nous credits block is agent-independent (a portal fetch), so it runs even
        with no live agent — important for the TUI, where /usage runs in a slash-worker
        subprocess that resumes the session WITHOUT building an agent (self.agent is None),
        which would otherwise early-return before any credits showed.
        """
        if not self.agent:
            if self._print_nous_credits_block():
                self._print_usage_cta()
            else:
                print("(._.) No active agent -- send a message first.")
            return

        agent = self.agent
        calls = agent.session_api_calls

        if calls == 0:
            if self._print_nous_credits_block():
                self._print_usage_cta()
            else:
                print("(._.) No API calls made yet in this session.")
            return

        # ── Rate limits (shown first when available) ────────────────
        rl_state = agent.get_rate_limit_state()
        if rl_state and rl_state.has_data:
            from agent.rate_limit_tracker import format_rate_limit_display
            print()
            print(format_rate_limit_display(rl_state))
            print()

        # ── Session token usage ─────────────────────────────────────
        input_tokens = getattr(agent, "session_input_tokens", 0) or 0
        output_tokens = getattr(agent, "session_output_tokens", 0) or 0
        reasoning_tokens = getattr(agent, "session_reasoning_tokens", 0) or 0
        prompt = agent.session_prompt_tokens
        completion = agent.session_completion_tokens
        total = agent.session_total_tokens

        compressor = agent.context_compressor
        last_prompt = compressor.last_prompt_tokens if compressor.last_prompt_tokens > 0 else 0
        ctx_len = compressor.context_length
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        compressions = compressor.compression_count

        msg_count = len(self.conversation_history)
        from cli import format_duration_compact
        elapsed = format_duration_compact((datetime.now() - self.session_start).total_seconds())

        print("  📊 Session Token Usage")
        print(f"  {'─' * 40}")
        print(f"  Model:                     {agent.model}")
        print(f"  Input tokens:              {input_tokens:>10,}")
        print(f"  Output tokens:             {output_tokens:>10,}")
        if reasoning_tokens:
            print(f"  ↳ Reasoning (subset):      {reasoning_tokens:>10,}")
        print(f"  Prompt tokens (total):     {prompt:>10,}")
        print(f"  Completion tokens:         {completion:>10,}")
        print(f"  Total tokens:              {total:>10,}")
        print(f"  API calls:                 {calls:>10,}")
        print(f"  Session duration:          {elapsed:>10}")
        print(f"  {'─' * 40}")
        print(f"  Current context:  {last_prompt:,} / {ctx_len:,} ({pct:.0f}%)")
        print(f"  Messages:         {msg_count}")
        print(f"  Compressions:     {compressions}")

        # Account limits -- fetched off-thread with a hard timeout so slow
        # provider APIs don't hang the prompt.
        provider = getattr(agent, "provider", None) or getattr(self, "provider", None)
        base_url = getattr(agent, "base_url", None) or getattr(self, "base_url", None)
        api_key = getattr(agent, "api_key", None) or getattr(self, "api_key", None)
        # Lazy import — pulls the OpenAI SDK chain, only needed here.
        from agent.account_usage import fetch_account_usage, render_account_usage_lines
        account_snapshot = None
        if provider:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                try:
                    account_snapshot = _pool.submit(
                        fetch_account_usage, provider,
                        base_url=base_url, api_key=api_key,
                    ).result(timeout=10.0)
                except (concurrent.futures.TimeoutError, Exception):
                    account_snapshot = None
        account_lines = [f"  {line}" for line in render_account_usage_lines(account_snapshot)]
        if account_lines:
            print()
            for line in account_lines:
                print(line)

        # Nous credits magnitudes + monthly-grant gauge (agent-independent — also
        # runs at the no-agent / no-calls early-returns above). See the helper.
        if self._print_nous_credits_block():
            self._print_usage_cta()

        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            for noisy in ('openai', 'openai._base_client', 'httpx', 'httpcore', 'asyncio', 'hpack', 'grpc', 'modal'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)
            # NOTE: We deliberately do NOT raise per-logger levels for
            # tools/run_agent/etc. in quiet mode. Setting logger.setLevel
            # above the file handler level filters records before they
            # reach handlers, so agent.log / errors.log lose visibility
            # into stream-retry events, credential rotations, etc.
            # Console quietness is enforced by hermes_logging not
            # installing a console StreamHandler in non-verbose mode.

    def _show_insights(self, command: str = "/insights"):
        """Show usage insights and analytics from session history."""
        # Parse optional --days flag
        parts = command.split()
        days = 30
        source = None
        i = 1
        while i < len(parts):
            if parts[i] == "--days" and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    print(f"  Invalid --days value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--source" and i + 1 < len(parts):
                source = parts[i + 1]
                i += 2
            elif parts[i].isdigit():
                days = int(parts[i])
                i += 1
            else:
                i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            db = SessionDB()
            engine = InsightsEngine(db)
            report = engine.generate(days=days, source=source)
            print(engine.format_terminal(report))
            db.close()
        except Exception as e:
            print(f"  Error generating insights: {e}")
