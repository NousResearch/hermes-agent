"""A viewer's ``/compress`` on a messaging chat (Telegram, Discord, ...), #144422.

``session.mutate`` ``compress`` names whatever row the viewer shows. A LOCAL session commits into its
frozen policy, so the mutation's prepare pass reads ``gateway.local_policy.v1:<session id>``; a
messaging chat is gateway-owned and has no such receipt, so the same mutation was refused with
``runtime_coordination_required`` before the summarizer ran — no compression attempt, no log line, an
unchanged transcript.

A messaging chat is compressed by the gateway that owns its lane, not by the mutation's own commit
path: the chat's ``/compress`` builds a throwaway agent, summarizes, and persists the result through
the session store (in-place compaction or rotation). This module routes the mutation there and answers
with the same report the chat prints, so Desktop, the TUI and the mobile app compress a messaging chat
exactly as the chat itself does.
"""

from hermes_state_runtime import RuntimeStoreError


def is_messaging_live(live) -> bool:
    """Whether *live* is a chat a messaging platform owns (anything but a LOCAL surface)."""
    from gateway.config import Platform

    return live is not None and live.source.platform != Platform.LOCAL


async def apply_chat_compress(authority, ref, live, params):
    """Run the chat's own ``/compress`` for a viewer mounted on *live* and answer with its report.

    The caller authorized control and holds the mount's ``claim_gate`` (a turn may not claim the chat
    while the summary runs); the compression core keeps its own durable lock and fences.
    """
    from gateway.session_mutation_compress import compress_request

    request = compress_request(params["payload"])
    runner = authority.runner
    store = runner.async_session_store
    # The lane's current conversation: the compression tip of the row the viewer named.
    entry = await store.get_or_create_session(live.source)
    previous = entry.session_id
    try:
        history = await store.load_transcript(previous)
    except Exception as exc:
        raise RuntimeStoreError("storage_unavailable", detail=str(exc)) from exc
    from agent.conversation_compression_manual import MIN_MESSAGES
    from agent.i18n import t

    if not history or len(history) < MIN_MESSAGES:
        return {
            "session_id": ref.session_id,
            "operation": "compress",
            "status": "noop",
            "lines": [t("gateway.compress.not_enough")],
        }
    messages = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]
    if request.preview:
        # Read-only: the report is the whole result; nothing is summarized or committed.
        from agent.model_metadata import estimate_request_tokens_rough
        from hermes_cli.partial_compress import summarize_compress_preview

        report = summarize_compress_preview(
            messages,
            request.partial,
            request.keep_last,
            request.focus_topic,
            estimate_request_tokens_rough(messages),
        )
        return {
            "session_id": ref.session_id,
            "operation": "compress",
            "status": "preview",
            "lines": report["lines"],
            "head_count": report["head_count"],
            "tail_count": report["tail_count"],
            "message_count": report["total"],
        }
    outcome = await runner._run_manual_compression(live.source, entry, history, request)
    if outcome.status == "lock_skipped":
        # Another writer holds the chat's compression lock: a definitive refusal the viewer retries.
        raise RuntimeStoreError("session_busy", detail=outcome.text)
    if outcome.status != "compressed":
        return {
            "session_id": ref.session_id,
            "operation": "compress",
            "status": "noop",
            "lines": [outcome.text],
        }
    target = (
        entry.session_id
    )  # rotation repointed the entry; in-place compaction kept it
    row = authority.db.get_session(target) or {}
    return {
        "session_id": ref.session_id,
        "operation": "compress",
        "status": "compressed",
        "target_session_id": target,
        "previous_target_session_id": previous,
        "revision": row.get("runtime_revision"),
        "message_count": row.get("message_count"),
        "lines": [outcome.text],
    }
