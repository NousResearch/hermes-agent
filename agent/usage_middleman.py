"""Compact renderer for the `/usage` command.

This module owns the fixed-width 79-character hash table used by the CLI and
gateway usage surfaces. It deliberately keeps rendering concerns separate from
provider fetching and session reconstruction logic:

- `cli.py` orchestrates the `/usage` command, including persisted-session
  fallback when no live agent exists yet.
- `agent/account_usage.py` fetches balance snapshots and provider-specific
  quota/account data.
- `build_compact_usage_table()` turns already-normalized session, balance, and
  quota inputs into a symmetric `#`-bordered table that never breaks the right
  edge.

The layout contract is intentionally strict so tests can assert every line is
exactly 79 characters wide.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import re
import textwrap

_TOTAL = 79
_INNER = _TOTAL - 2
_COL1 = 18
_COL2 = 54


def _clamp_pct(pct: float | int | None) -> float:
    try:
        value = float(pct or 0.0)
    except Exception:
        value = 0.0
    return max(0.0, min(100.0, value))


def _compact_bar(pct: float | int | None, width: int = 16) -> str:
    pct_value = _clamp_pct(pct)
    filled = int((pct_value / 100.0) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _bar() -> str:
    return "#" * _TOTAL


def _divider() -> str:
    return "#" + "-" * _INNER + "#"


def _center_line(text: str) -> str:
    title_text = f" {text.strip()} " if text.strip() else " "
    if len(title_text) > _INNER:
        title_text = " " + title_text.strip()[: _INNER - 4] + "… "
    return f"#{title_text:^{_INNER}}#"


def _wrap_value(value: str, width: int) -> list[str]:
    text = str(value or "")
    if not text:
        return [""]
    wrapped = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        drop_whitespace=False,
    )
    return wrapped or [text[:width]]


def _row(label: str, value: str) -> list[str]:
    lines: list[str] = []
    chunks = _wrap_value(value, _COL2)
    for index, chunk in enumerate(chunks):
        shown_label = label if index == 0 else ""
        lines.append(f"# {shown_label:<{_COL1}} | {chunk:<{_COL2}} #")
    return lines


def _append_section(lines: list[str], title: str, rows: Iterable[tuple[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    if len(lines) > 1:
        lines.append(_divider())
    lines.append(_center_line(title))
    for label, value in rows:
        lines.extend(_row(label, value))


def _center_content_row(text: str) -> str:
    shown = str(text or "").strip()
    if len(shown) > _INNER:
        shown = shown[: _INNER - 1] + "…"
    return f"#{shown:^{_INNER}}#"


def _extract_metric(value: str, pattern: str) -> str | None:
    match = re.search(pattern, value)
    if not match:
        return None
    return match.group(1).strip()


def _cell_grid(cells: Sequence[tuple[str, int]]) -> list[str]:
    border = "+" + "+".join("-" * (width + 2) for _, width in cells) + "+"
    content = "|" + "|".join(f" {text:^{width}} " for text, width in cells) + "|"
    return [border, content, border]


def _format_balance_lines(label: str, value: str) -> list[str]:
    provider = str(label or "").strip().lower()
    detail = str(value or "").strip()

    if provider == "openrouter":
        balance = _extract_metric(detail, r"Credits balance:\s*([^\s•]+)")
        total = _extract_metric(detail, r"API key usage:\s*([^\s]+)\s+total")
        today = _extract_metric(detail, r"([^\s]+)\s+today")
        week = _extract_metric(detail, r"([^\s]+)\s+this week")
        month = _extract_metric(detail, r"([^\s]+)\s+this month")
        top_cells = []
        bottom_cells = []
        if today:
            top_cells.append((f"d {today}", 8))
        if week:
            top_cells.append((f"w {week}", 8))
        if month:
            top_cells.append((f"m {month}", 8))
        if balance:
            bottom_cells.append((f"bal {balance}", 10))
        if total:
            bottom_cells.append((f"Σ {total}", 8))
        metric_lines: list[str] = ["openrouter"]
        if top_cells:
            metric_lines.extend(_cell_grid(top_cells))
        if bottom_cells:
            metric_lines.extend(_cell_grid(bottom_cells))
        if len(metric_lines) > 1:
            return metric_lines

    if provider == "maritaca":
        amount = _extract_metric(detail, r"Saldo:\s*(R\$\s*[0-9.,]+)")
        if amount:
            return ["maritaca", f"saldo {amount}"]

    if provider:
        return [f"{provider}  {detail}"]
    return [detail]


def _append_balance_section(lines: list[str], rows: Iterable[tuple[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    if len(lines) > 1:
        lines.append(_divider())
    lines.append(_center_line("BALANCES"))
    for label, value in rows:
        lines.append(_divider())
        for balance_line in _format_balance_lines(label, value):
            lines.append(_center_content_row(balance_line))


def _format_session_rows(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    total_tokens: int,
    cost_usd: float | None,
    cost_status: str,
    duration_str: str,
    context_tokens: int,
    context_length: int,
    api_calls: int,
    message_count: int | None,
    compression_count: int | None,
    session_notes: Sequence[str],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    token_main = f"in {input_tokens:,}  out {output_tokens:,}  total {total_tokens:,}"
    cache_parts = []
    if cache_read_tokens:
        cache_parts.append(f"cache-r {cache_read_tokens:,}")
    if cache_write_tokens:
        cache_parts.append(f"cache-w {cache_write_tokens:,}")
    if cache_parts:
        candidate = token_main + "  " + "  ".join(cache_parts)
        if len(candidate) <= _COL2:
            rows.append(("tokens", candidate))
        else:
            rows.append(("tokens", token_main))
            rows.append(("cache", "  ".join(cache_parts)))
    else:
        rows.append(("tokens", token_main))

    if cost_status == "included":
        cost_value = "included"
    elif cost_usd is not None:
        prefix = "~" if cost_status == "estimated" else ""
        cost_value = f"{prefix}${float(cost_usd):.4f} est."
    else:
        cost_value = "n/a"
    if duration_str:
        cost_value += f"  {duration_str}"
    rows.append(("cost", cost_value))

    if context_length > 0:
        used_pct = min(100.0, (float(context_tokens) / float(context_length)) * 100.0)
        context_value = f"{_compact_bar(used_pct)} {round(used_pct):.0f}%  {context_tokens:,} / {context_length:,}"
    else:
        context_value = f"{_compact_bar(0)} 0%  {context_tokens:,} / unknown"
    if duration_str:
        context_value += f"  {duration_str}"
    rows.append(("context", context_value))

    calls_value = f"{api_calls}"
    extra_parts = []
    if message_count is not None:
        extra_parts.append(f"msgs {message_count}")
    if compression_count:
        extra_parts.append(f"compressions {compression_count}")
    if extra_parts:
        calls_value += "  " + "  ".join(extra_parts)
    rows.append(("calls", calls_value))

    for note in session_notes:
        rows.append(("note", note))
    return rows


def _format_quota_row(label: str, pct_used: float | int | None, reset_text: str) -> tuple[str, str]:
    pct_left = max(0, round(100 - _clamp_pct(pct_used)))
    detail = (reset_text or "unknown").strip() or "unknown"
    if not detail.startswith("resets "):
        detail = f"resets {detail}"
    return label, f"{_compact_bar(pct_left)} {pct_left}% left  {detail}"


def build_compact_usage_table(
    *,
    model: str,
    provider: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    total_tokens: int,
    cost_usd: float | None,
    cost_status: str,
    duration_str: str,
    context_tokens: int,
    context_length: int,
    api_calls: int,
    balance_rows: Sequence[tuple[str, str]] | None = None,
    quota_sections: Sequence[tuple[str, Sequence[tuple[str, float | int | None, str]]]] | None = None,
    include_session: bool = True,
    message_count: int | None = None,
    compression_count: int | None = None,
    session_notes: Sequence[str] | None = None,
) -> list[str]:
    title = f"Usage  {provider} / {model}" if provider else f"Usage  {model}"
    lines = [_bar(), _center_line(title)]

    if include_session:
        session_rows = _format_session_rows(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_status=cost_status,
            duration_str=duration_str,
            context_tokens=context_tokens,
            context_length=context_length,
            api_calls=api_calls,
            message_count=message_count,
            compression_count=compression_count,
            session_notes=tuple(session_notes or ()),
        )
        _append_section(lines, "session", session_rows)

    if balance_rows:
        _append_balance_section(lines, balance_rows)

    for section_title, section_rows in quota_sections or ():
        if not section_rows:
            continue
        quota_rows = [_format_quota_row(label, pct_used, reset_text) for label, pct_used, reset_text in section_rows]
        _append_section(lines, section_title, quota_rows)

    lines.append(_bar())
    return lines
