"""
Canonical list of OpenRouter models offered in CLI and setup wizards.

Add, remove, or reorder entries here — both `hermes setup` and
`hermes` provider-selection will pick up the change automatically.
"""

# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("anthropic/claude-opus-4.6",       "recommended"),
    ("anthropic/claude-sonnet-4.5",     ""),
    ("anthropic/claude-opus-4.5",       ""),
    ("openai/gpt-5.2",                  ""),
    ("openai/gpt-5.3-codex",            ""),
    ("google/gemini-3-pro-preview",     ""),
    ("google/gemini-3-flash-preview",   ""),
    ("z-ai/glm-4.7",                    ""),
    ("moonshotai/kimi-k2.5",            ""),
    ("minimax/minimax-m2.1",            ""),
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

# Chutes.ai models (HuggingFace-style slugs, fetched live via /v1/models when key is set)
CHUTES_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen3-32B", ""),
    ("deepseek-ai/DeepSeek-V3-0324-TEE", ""),
    ("unsloth/Mistral-Nemo-Instruct-2407", ""),
    ("unsloth/gemma-3-27b-it", ""),
    ("deepseek-ai/DeepSeek-V3.2-TEE", ""),
    ("openai/gpt-oss-120b-TEE", ""),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507-TEE", ""),
    ("zai-org/GLM-4.7-TEE", ""),
    ("chutesai/Mistral-Small-3.1-24B-Instruct-2503", ""),
    ("moonshotai/Kimi-K2.5-TEE", ""),
    ("zai-org/GLM-5-TEE", ""),
    ("deepseek-ai/DeepSeek-R1-0528-TEE", ""),
    ("chutesai/Mistral-Small-3.2-24B-Instruct-2506", ""),
    ("OpenGVLab/InternVL3-78B-TEE", ""),
    ("deepseek-ai/DeepSeek-V3.1-TEE", ""),
    ("XiaomiMiMo/MiMo-V2-Flash-TEE", ""),
    ("Qwen/Qwen3-Coder-Next-TEE", ""),
    ("MiniMaxAI/MiniMax-M2.5-TEE", ""),
    ("deepseek-ai/DeepSeek-V3.1-Terminus-TEE", ""),
    ("NousResearch/Hermes-4-405B-FP8-TEE", ""),
    ("tngtech/DeepSeek-TNG-R1T2-Chimera", ""),
    ("zai-org/GLM-4.6-TEE", ""),
    ("Qwen/Qwen3-235B-A22B-Thinking-2507", ""),
    ("Qwen/Qwen3.5-397B-A17B-TEE", ""),
    ("deepseek-ai/DeepSeek-V3", ""),
    ("zai-org/GLM-4.6V", ""),
    ("Qwen/Qwen2.5-VL-72B-Instruct-TEE", ""),
    ("zai-org/GLM-4.6-FP8", ""),
    ("zai-org/GLM-4.7-FP8", ""),
    ("openai/gpt-oss-20b", ""),
    ("tngtech/R1T2-Chimera-Speed", ""),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", ""),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", ""),
    ("Qwen/Qwen3-30B-A3B", ""),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", ""),
    ("Qwen/Qwen2.5-72B-Instruct", ""),
    ("Qwen/Qwen2.5-VL-32B-Instruct", ""),
    ("deepseek-ai/DeepSeek-V3.2-Speciale-TEE", ""),
    ("Qwen/Qwen3-VL-235B-A22B-Instruct", ""),
    ("unsloth/Mistral-Small-24B-Instruct-2501", ""),
    ("miromind-ai/MiroThinker-v1.5-235B", ""),
    ("unsloth/gemma-3-12b-it", ""),
    ("Qwen/Qwen3-14B", ""),
    ("Qwen/Qwen2.5-Coder-32B-Instruct", ""),
    ("NousResearch/Hermes-4-14B", ""),
    ("unsloth/gemma-3-4b-it", ""),
    ("NousResearch/DeepHermes-3-Mistral-24B-Preview", ""),
    ("unsloth/Llama-3.2-1B-Instruct", ""),
    ("rednote-hilab/dots.ocr", ""),
    ("Qwen/Qwen3Guard-Gen-0.6B", ""),
    ("unsloth/Llama-3.2-3B-Instruct", ""),
]
