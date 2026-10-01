"""CLI formatting of canonical per-token prices."""

def _format_price_per_mtok(per_token_str: str) -> str:
    """Per-token price string → $/Mtok string. Always 2 decimals so right-justified prices align;
    sub-cent prices (deep-discount cache-hit promos) widen precision until the value shows, keep
    one extra digit and trim trailing zeros instead of collapsing to "$0.00"."""
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    per_m = val * 1_000_000
    text = f"{per_m:.2f}"
    if per_m < 0.01:
        prec = 3
        while prec < 12 and round(per_m, prec) == 0:
            prec += 1
        text = f"{per_m:.{min(prec + 1, 12)}f}".rstrip("0").rstrip(".")
    return f"${text}"
