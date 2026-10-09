"""Display-only provider grouping for interactive application surfaces."""

from __future__ import annotations

PROVIDER_GROUPS: dict[str, tuple[str, str, list[str]]] = {
    "kimi": ("Kimi / Moonshot", "Coding Plan, Moonshot global & China endpoints", ["kimi-coding", "kimi-coding-cn"]),
    "minimax": ("MiniMax", "Global, OAuth Coding Plan & China endpoints", ["minimax", "minimax-oauth", "minimax-cn"]),
    "xai": ("xAI Grok", "Direct API or SuperGrok / Premium+ OAuth", ["xai", "xai-oauth"]),
    "google": ("Google Gemini", "Google AI Studio (API key)", ["gemini"]),
    "openai": ("OpenAI", "ChatGPT/Codex subscription or direct OpenAI API", ["openai-codex", "openai-api"]),
    "qwen": ("Qwen", "Qwen Cloud / DashScope, Coding Plan, Token Plan & Qwen CLI OAuth", ["alibaba", "alibaba-cn", "alibaba-coding-plan", "alibaba-coding-plan-cn", "alibaba-token-plan", "alibaba-token-plan-cn", "qwen-oauth"]),
    "opencode": ("OpenCode", "Zen pay-as-you-go or Go subscription", ["opencode-zen", "opencode-go"]),
    "copilot": ("GitHub Copilot", "GitHub token API or copilot --acp process", ["copilot", "copilot-acp"]),
    "tencent": ("Tencent Hy", "Hy4 / Hy3 via TokenHub & TokenPlan", ["tencent-tokenhub", "tencent-tokenplan"]),
}

_SLUG_TO_GROUP: dict[str, str] = {
    slug: group_id
    for group_id, (_label, _description, members) in PROVIDER_GROUPS.items()
    for slug in members
}


def provider_group_for_slug(slug: str) -> str:
    return _SLUG_TO_GROUP.get(str(slug or "").strip().lower(), "")


def group_providers(slugs):
    """Fold ordered provider slugs into display-only picker rows."""
    present = set(slugs)
    group_members = {
        group_id: [member for member in members if member in present]
        for group_id, (_label, _description, members) in PROVIDER_GROUPS.items()
    }
    rows = []
    seen: set[str] = set()
    emitted_groups: set[str] = set()
    for slug in slugs:
        normalized = str(slug or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        group_id = _SLUG_TO_GROUP.get(normalized, "")
        if not group_id:
            rows.append({"kind": "single", "slug": normalized})
            continue
        if group_id in emitted_groups:
            continue
        emitted_groups.add(group_id)
        members = group_members.get(group_id) or [normalized]
        if len(members) <= 1:
            rows.append({"kind": "single", "slug": members[0]})
        else:
            label, description, _ = PROVIDER_GROUPS[group_id]
            rows.append({
                "kind": "group",
                "group_id": group_id,
                "label": label,
                "description": description,
                "members": list(members),
            })
    return rows


__all__ = ["PROVIDER_GROUPS", "group_providers", "provider_group_for_slug"]


def provider_label(provider: str | None) -> str:
    """Return the effective provider declaration's display label."""
    original = (provider or "openrouter").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    from providers import get_provider_profile

    profile = get_provider_profile(normalized)
    return str(profile.display_name or profile.name) if profile else (original or "OpenRouter")
