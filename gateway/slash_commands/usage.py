"""/usage, /topup, /insights slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

import asyncio
import re

from agent.account_usage import fetch_account_usage
from agent.account_usage import render_account_usage_lines
from agent.i18n import t
from gateway.platforms.base import MessageEvent

from gateway.slash_commands._shared import logger

class UsageCommandsMixin:
    """/usage, /topup, /insights handlers."""

    async def _handle_topup_command(self, event: MessageEvent) -> str:
        """Handle /topup -- show the Nous balance and hand off to the portal.

        Renders the balance block + identity line + a tappable portal URL that
        opens the billing page. Remote spending is managed on the portal: this
        messaging command does NOT charge, confirm, or track payment here —
        everything happens in the browser and the next /topup shows the new balance. The
        tappable URL is the affordance and works on every platform (button-capable
        or plain text like SMS/email). Fetched off the event loop; fail-open.
        """
        from agent.account_usage import build_credits_view

        try:
            view = await asyncio.to_thread(build_credits_view, markdown=True)
        except Exception:
            view = None

        if view is None or not view.logged_in:
            return t("gateway.credits.not_logged_in")

        lines: list[str] = ["💳 **Nous balance**"]
        for line in view.balance_lines:
            if line.lstrip().startswith("📈"):
                continue  # drop the helper's header; we print our own
            lines.append(line)
        if view.identity_line:
            lines.append("")
            lines.append(view.identity_line)
        if view.topup_url:
            lines.append("")
            lines.append(f"Manage billing on the portal: {view.topup_url}")
            lines.append("Top up and manage billing in the browser — your balance updates here after.")
        return "\n".join(lines)

    def _context_breakdown_lines(self, agent, source) -> list[str]:
        """Render the per-category context breakdown for /usage.

        Estimated (chars/4) — same engine the desktop popover uses. Returns an
        empty list and never raises on failure so /usage stays robust.
        """
        try:
            from agent.context_breakdown import compute_session_context_breakdown

            history: list[dict] = []
            try:
                entry = self.session_store.get_or_create_session(source)
                history = self.session_store.load_transcript(entry.session_id) or []
            except Exception:
                history = []

            payload = compute_session_context_breakdown(agent, history)
            categories = payload.get("categories") or []
            if not categories:
                return []

            total = payload.get("estimated_total") or 0
            out = [t("gateway.usage.breakdown_header")]
            for cat in categories:
                tokens = int(cat.get("tokens") or 0)
                if tokens <= 0:
                    continue
                cat_id = str(cat.get("id") or "")
                label = t(f"gateway.usage.breakdown_cat_{cat_id}")
                # Missing key → t() echoes the key back; fall back to the
                # English label the engine already provides.
                if label.endswith(f"breakdown_cat_{cat_id}"):
                    label = str(cat.get("label") or cat_id)
                pct = round(tokens / total * 100) if total else 0
                out.append(
                    t("gateway.usage.breakdown_line", label=label, count=f"{tokens:,}", pct=pct)
                )
            return out if len(out) > 1 else []
        except Exception:
            return []

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Checks both _running_agents (mid-turn) and _agent_cache (between turns)
        so that rate limits, cost estimates, and detailed token breakdowns are
        available whenever the user asks, not only while the agent is running.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        source = event.source
        session_key = self._session_key_for_source(source)

        # `/usage reset [--force]` — redeem one banked Codex rate-limit reset
        # credit. Parsed before the display path so it never mixes with the
        # stats rendering below.
        raw_args = event.get_command_args().strip()
        args = [a.lower() for a in raw_args.split()] if raw_args else []
        wants_reset = bool(args) and args[0] == "reset"
        if args and not wants_reset:
            return t("gateway.usage.unknown_subcommand", args=raw_args)

        # Try running agent first (mid-turn), then cached agent (between turns)
        agent = self._running_agents.get(session_key)
        if not agent or agent is _AGENT_PENDING_SENTINEL:
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    cached = _cache.get(session_key)
                    if cached:
                        agent = cached[0]

        # Resolve provider/base_url/api_key for the account-usage fetch.
        # Prefer the live agent; fall back to persisted billing data on the
        # SessionDB row so `/usage` still returns account info between turns
        # when no agent is resident.
        provider = getattr(agent, "provider", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
        base_url = getattr(agent, "base_url", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
        api_key = getattr(agent, "api_key", None) if agent and agent is not _AGENT_PENDING_SENTINEL else None
        if not provider and getattr(self, "_session_db", None) is not None:
            try:
                _entry_for_billing = await self.async_session_store.get_or_create_session(source)
                persisted = await self._session_db.get_session(_entry_for_billing.session_id) or {}
            except Exception:
                persisted = {}
            provider = provider or persisted.get("billing_provider")
            base_url = base_url or persisted.get("billing_base_url")

        if wants_reset:
            normalized_provider = str(provider or "").strip().lower()
            if normalized_provider != "openai-codex":
                return t("gateway.usage.reset_wrong_provider")
            force = "--force" in args[1:]
            from agent.account_usage import redeem_codex_reset_credit

            result = await asyncio.to_thread(
                redeem_codex_reset_credit,
                base_url=base_url,
                api_key=api_key,
                force=force,
            )
            return result.message

        # Fetch account usage off the event loop so slow provider APIs don't
        # block the gateway. Failures are non-fatal -- account_lines stays [].
        account_lines: list[str] = []
        credits_lines: list[str] = []
        if provider:
            try:
                account_snapshot = await asyncio.to_thread(
                    fetch_account_usage,
                    provider,
                    base_url=base_url,
                    api_key=api_key,
                )
            except Exception:
                account_snapshot = None
            if account_snapshot:
                account_lines = render_account_usage_lines(account_snapshot, markdown=True)

        # ── Nous credits magnitudes + monthly-grant % gauge ─────────────
        # Shared with the CLI / TUI /usage block via nous_credits_lines(): a single
        # auth-gate + portal-fetch + render path (which also honors the dev fixture).
        # Run off the event loop. The helper gates on "a Nous account is logged in"
        # — NOT the inference provider and NOT nested under `if provider:` — so a
        # Nous-credentialled user running inference elsewhere (or with none resident)
        # still sees their balance. NO recovery trigger: messaging binds no notice
        # consumer, so /usage only displays. Fail-open: never break /usage.
        try:
            from agent.account_usage import nous_credits_lines

            credits_lines = await asyncio.to_thread(nous_credits_lines, markdown=True)
        except Exception:
            credits_lines = []  # fail-open: never break /usage

        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = []

            # Rate limits (when available from provider headers)
            rl_state = agent.get_rate_limit_state()
            if rl_state and rl_state.has_data:
                from agent.rate_limit_tracker import format_rate_limit_compact
                lines.append(t("gateway.usage.rate_limits", state=format_rate_limit_compact(rl_state)))
                lines.append("")

            # Session token usage — detailed breakdown matching CLI
            input_tokens = getattr(agent, "session_input_tokens", 0) or 0
            output_tokens = getattr(agent, "session_output_tokens", 0) or 0

            lines.append(t("gateway.usage.header_session"))
            lines.append(t("gateway.usage.label_model", model=agent.model))
            lines.append(t("gateway.usage.label_input_tokens", count=f"{input_tokens:,}"))
            lines.append(t("gateway.usage.label_output_tokens", count=f"{output_tokens:,}"))
            lines.append(t("gateway.usage.label_total", count=f"{agent.session_total_tokens:,}"))
            lines.append(t("gateway.usage.label_api_calls", count=agent.session_api_calls))

            # Context window and compressions
            ctx = agent.context_compressor
            _lpt = ctx.last_prompt_tokens if ctx.last_prompt_tokens > 0 else 0
            if _lpt:
                pct = min(100, _lpt / ctx.context_length * 100) if ctx.context_length else 0
                lines.append(t("gateway.usage.label_context", used=f"{_lpt:,}", total=f"{ctx.context_length:,}", pct=f"{pct:.0f}"))
            if ctx.compression_count:
                lines.append(t("gateway.usage.label_compressions", count=ctx.compression_count))

            # Per-category context breakdown (estimated — chars/4 heuristic).
            # Same engine the desktop popover uses (PR #54907). The system
            # prompt / tools / skills / memory slices read off the live agent;
            # the conversation slice is estimated from the session transcript.
            breakdown_lines = await asyncio.to_thread(
                self._context_breakdown_lines, agent, source
            )
            if breakdown_lines:
                lines.append("")
                lines.extend(breakdown_lines)

            if account_lines:
                lines.append("")
                lines.extend(account_lines)
            if credits_lines:
                lines.append("")
                lines.extend(credits_lines)

            return "\n".join(lines)

        # No agent at all -- check session history for a rough count
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            lines = [
                t("gateway.usage.header_session_info"),
                t("gateway.usage.label_messages", count=len(msgs)),
                t("gateway.usage.label_estimated_context", count=f"{approx:,}"),
                t("gateway.usage.detailed_after_first"),
            ]
            if account_lines:
                lines.append("")
                lines.extend(account_lines)
            if credits_lines:
                lines.append("")
                lines.extend(credits_lines)
            return "\n".join(lines)
        if account_lines or credits_lines:
            # account-only, credits-only, or both — joined with a blank divider.
            parts = list(account_lines)
            if credits_lines:
                if parts:
                    parts.append("")
                parts.extend(credits_lines)
            return "\n".join(parts)
        return t("gateway.usage.no_data")

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        args = event.get_command_args().strip()

        # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
        args = re.sub(r'[\u2012\u2013\u2014\u2015](days|source)', r'--\1', args)

        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return t("gateway.insights.invalid_days", value=parts[i + 1])
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

            loop = asyncio.get_running_loop()

            def _run_insights():
                db = SessionDB()
                engine = InsightsEngine(db)
                report = engine.generate(days=days, source=source)
                result = engine.format_gateway(report)
                db.close()
                return result

            return await loop.run_in_executor(None, _run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return t("gateway.insights.error", error=e)
