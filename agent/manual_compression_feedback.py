"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
) -> dict[str, Any]:
    """Return consistent user-facing feedback for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)

    if noop:
        headline = f"圧縮による変更はありません: {before_count}件のメッセージ"
        if after_tokens == before_tokens:
            token_line = (
                f"概算リクエストサイズ: 約{before_tokens:,}トークン（変更なし）"
            )
        else:
            token_line = (
                f"概算リクエストサイズ: 約{before_tokens:,} → "
                f"約{after_tokens:,}トークン"
            )
    else:
        headline = f"圧縮しました: {before_count} → {after_count}件のメッセージ"
        token_line = (
            f"概算リクエストサイズ: 約{before_tokens:,} → "
            f"約{after_tokens:,}トークン"
        )

    note = None
    if not noop and after_count < before_count and after_tokens > before_tokens:
        note = (
            "補足: メッセージ数が減っても、要約が高密度になると概算サイズが増える場合があります。"
        )

    return {
        "noop": noop,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
