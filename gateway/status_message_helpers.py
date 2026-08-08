"""Status-message helpers extracted from gateway/run.py (slice 14 of #54962).

Pure helpers that shape status messages for messaging-platform delivery.
Extracted verbatim — zero behavior change. ``gateway.run`` re-imports these
names so ``gateway.run.<name>`` references (and existing tests) keep working.

NOTE — ``_prepare_gateway_status_message`` remains in gateway/run.py on
purpose: it reads ``_TELEGRAM_NOISY_STATUS_RE`` /
``_gateway_compression_progress_notices_enabled`` /
``_gateway_surface_passes_raw_text`` (kept in run.py), the redaction helpers
``_redact_gateway_user_facing_secrets`` / ``_looks_like_gateway_provider_error``
/ ``_gateway_provider_error_reply`` (extracted in parallel by #77707), and
``_COMPRESSION_PROGRESS_STATUS_RE`` (extracted by #77450). It moves here once
those land and the imports can be made without circularity.
"""


def render_notice_line(notice) -> str:
    """Render an AgentNotice to a single plaintext line for messaging platforms.

    Messaging has no persistent status bar (unlike the TUI), so a notice is a
    one-shot standalone push. The notice policy already bakes the level glyph
    (⚠ / • / ✕ / ✓) into the text, and the TUI + CLI REPL render that text
    verbatim — so we emit it as-is here too. Prepending a per-level glyph would
    DOUBLE it ("⚠ ⚠ Credits 90% used", "⛔ ✕ Credit access paused"). Plaintext
    only — no markdown — so it renders uniformly across Telegram/Discord/Slack/
    SMS without per-platform escaping. Fail-soft: a malformed/empty notice
    degrades to "" rather than raising on the agent's callback path.
    """
    return str(getattr(notice, "text", "") or "").strip()


async def _send_or_update_status_coro(adapter, chat_id, status_key, content, metadata):
    """Route a status message through adapter.send_or_update_status when supported.

    Issue #30045: adapters that implement send_or_update_status (currently
    Telegram) edit the previous bubble for the same status_key instead of
    appending a new one. Adapters without the method fall back to plain send.
    """
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)
