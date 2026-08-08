"""Nicholas's curated cross-provider model picker entries."""

MY_MODELS = [
    # Keep subscription and paid API routes visibly separate.
    ("Luna Sub", "openai-codex", "gpt-5.6-luna"),
    ("Sol Sub", "openai-codex", "gpt-5.6-sol"),
    ("Luna Paid API (direct)", "openai", "gpt-5.6-luna"),
    ("Luna Pro API (direct)", "openai", "gpt-5.6-luna-pro"),
    ("GPT-5.4 (direct API)", "openai", "gpt-5.4"),
    ("GPT-5.4 Mini (direct API)", "openai", "gpt-5.4-mini"),
    ("Gemini 3.5 Flash (direct API)", "google", "gemini-3.5-flash"),
    ("Gemini 3.1 Flash Lite (direct, experimental tools)", "google", "gemini-3.1-flash-lite"),

    # OpenRouter-paid shortlist: current, capable, and useful for agent work.
    ("GLM 5.2", "openrouter", "z-ai/glm-5.2"),
    ("Kimi K3", "openrouter", "moonshotai/kimi-k3"),
    ("Qwen 3.7 Max", "openrouter", "qwen/qwen3.7-max"),
    ("Qwen 3.8 Max", "openrouter", "qwen/qwen3.8-max"),
    ("Grok 4.5", "openrouter", "x-ai/grok-4.5"),
    ("MiniMax M3", "openrouter", "minimax/minimax-m3"),
    ("MiMo V2.5 Pro", "openrouter", "xiaomi/mimo-v2.5-pro"),
]

# Internal picker IDs remain opaque to the UI but preserve the real provider.
# The delimiter is deliberately uncommon in model IDs.
_PREFIX = "my-model::"


def picker_models() -> list[str]:
    return [f"{_PREFIX}{provider}::{model}" for _, provider, model in MY_MODELS]


def display_name(picker_id: str) -> str:
    for label, provider, model in MY_MODELS:
        if picker_id == f"{_PREFIX}{provider}::{model}":
            return label
    return picker_id


def resolve(picker_id: str) -> tuple[str, str] | None:
    if not picker_id.startswith(_PREFIX):
        return None
    value = picker_id[len(_PREFIX):]
    provider, separator, model = value.partition("::")
    if not separator or not provider or not model:
        return None
    return provider, model
