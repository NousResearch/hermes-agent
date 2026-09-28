"""Telegram /model picker: collapse the picker payload to three providers.

Opt-in via ``model_catalog.telegram_three_provider_menu: true``. The classic
picker adds every authenticated built-in (MoA, DeepSeek, Ollama Cloud, ...) and
surfaces OpenRouter twice (built-in ``openrouter`` + ``custom:openrouter``).
This helper rewrites that payload to exactly:

    opencode-zen, opencode-go, openrouter

Alias rows (e.g. ``custom:openrouter``) contribute their live model catalog but
never their own identity: the canonical row's slug/name are kept so the model
switch keeps routing through the provider the runtime understands.
"""
from __future__ import annotations

# canonical slug -> aliases that must fold into it (lower-case slug compare).
# ``custom:<name>`` rows are treated as aliases of ``<name>``.
_CANONICAL: dict[str, tuple[str, ...]] = {
    "opencode-zen": ("opencode-zen",),
    "opencode-go": ("opencode-go",),
    "openrouter": ("openrouter",),
}


def _canonical_for(slug: str) -> str | None:
    """Map a picker slug (including ``custom:<name>``) to its canonical slug."""
    slug = (slug or "").strip().lower()
    if slug.startswith("custom:"):
        slug = slug.split(":", 1)[1].strip()
    for canonical, aliases in _CANONICAL.items():
        if slug in aliases:
            return canonical
    return None


def three_provider_menu_enabled() -> bool:
    """True when ``model_catalog.telegram_three_provider_menu`` is set.

    Opt-in: with the flag unset the classic grouped picker behaves exactly as
    before. Read fails closed (unreadable/absent config → disabled).
    """
    try:
        import os

        import yaml

        home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        with open(os.path.join(home, "config.yaml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return bool((cfg.get("model_catalog") or {}).get("telegram_three_provider_menu"))
    except Exception:
        return False


_MDV2_SPECIALS = r"_*[]()~`>#+-=|{}.!"


def escape_md(text: str) -> str:
    """Escape Telegram MarkdownV2 specials so user text can't break the send."""
    out = []
    for ch in str(text):
        out.append("\\" + ch if ch in _MDV2_SPECIALS else ch)
    return "".join(out)

_DISPLAY_NAMES = {
    "opencode-zen": "OpenCode Zen",
    "opencode-go": "OpenCode Go",
    "openrouter": "OpenRouter",
}


def build_three_provider_payload(providers: list) -> list:
    """Return only the three canonical providers, with merged model catalogs."""
    rows = [p for p in (providers or []) if isinstance(p, dict) and p.get("slug")]

    by_canonical: dict[str, list] = {}
    for row in rows:
        canonical = _canonical_for(str(row.get("slug", "")))
        if canonical:
            by_canonical.setdefault(canonical, []).append(row)

    out: list = []
    for canonical in ("opencode-zen", "opencode-go", "openrouter"):
        members = by_canonical.get(canonical) or []
        if not members:
            continue

        # Canonical built-in row wins for identity/metadata; fall back to the
        # first alias row (a user-defined endpoint) if no exact slug is present.
        primary = next(
            (m for m in members if str(m.get("slug", "")).lower() == canonical),
            members[0],
        )

        merged: list = []
        for member in members:
            for model_id in member.get("models") or []:
                model_id = str(model_id)
                if model_id and model_id not in merged:
                    merged.append(model_id)
        if not merged:
            continue

        row = dict(primary)
        row["slug"] = canonical
        row["name"] = primary.get("name") or _DISPLAY_NAMES[canonical]
        row["models"] = merged
        row["total_models"] = len(merged)
        # Never advertise an alias row as user-defined: the picker treats such
        # rows as free-form endpoints and would bypass canonical routing.
        row["is_user_defined"] = False
        out.append(row)

    return out