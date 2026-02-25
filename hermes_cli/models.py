"""
Canonical list of OpenRouter models offered in CLI and setup wizards.

Add, remove, or reorder entries here — both `hermes setup` and
`hermes` provider-selection will pick up the change automatically.
"""

# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("anthropic/claude-opus-4.6", "recommended"),
    ("anthropic/claude-sonnet-4.5", ""),
    ("anthropic/claude-opus-4.5", ""),
    ("openai/gpt-5.2", ""),
    ("openai/gpt-5.3-codex", ""),
    ("google/gemini-3-pro-preview", ""),
    ("google/gemini-3-flash-preview", ""),
    ("z-ai/glm-4.7", ""),
    ("moonshotai/kimi-k2.5", ""),
    ("minimax/minimax-m2.1", ""),
]


def model_ids() -> list[str]:
    """Return just the model-id strings (convenience helper)."""
    return [mid for mid, _ in OPENROUTER_MODELS]


def menu_labels() -> list[str]:
    """Return display labels like 'anthropic/claude-opus-4.6 (recommended)'."""
    labels = []
    for mid, desc in OPENROUTER_MODELS:
        labels.append(f"{mid} ({desc})" if desc else mid)
    return labels
