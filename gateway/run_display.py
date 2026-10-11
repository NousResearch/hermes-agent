"""Small display-formatting helpers shared by gateway run phases."""


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _bg_prompt_preview(prompt: str, limit: int = 60) -> str:
    """Short single-line quote of a /bg prompt for its failure notice."""
    text = " ".join(str(prompt or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"