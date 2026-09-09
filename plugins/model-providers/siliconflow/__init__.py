"""SiliconFlow / 硅基流动 provider profiles (international + China).

SiliconFlow runs two **fully independent** platforms. They share no accounts,
no API keys, no model catalog and no base URL:

  * 国际站 (SiliconFlow)    https://api.siliconflow.com/v1   -> ``siliconflow``
  * 中文站 (硅基流动 / CN)   https://api.siliconflow.cn/v1    -> ``siliconflow-cn``

Both speak the OpenAI-compatible Chat Completions wire, so the shared
``openai.OpenAI`` client works unmodified and the inherited
``ProviderProfile.fetch_models()`` (GET ``{base_url}/models`` with Bearer auth)
picks up the live catalog for free.

Model ids are ``vendor/model`` slugs — ``deepseek-ai/DeepSeek-V4-Flash``,
``Qwen/Qwen3.6-27B``, ``zai-org/GLM-5.2`` — which
``normalize_model_for_provider()`` passes through untouched on its default
fall-through branch, so no per-model name repair is needed. The one case that
matters is a user pasting ``siliconflow/<model>`` (the config-file form) into
``/model``; that is repaired by the ``_MATCHING_PREFIX_STRIP_PROVIDERS`` entry
in ``hermes_cli/model_normalize.py``, which strips the prefix only when it
resolves to this provider — ``deepseek-ai/…`` is never mangled.

Slug naming follows the repo's convention for split CN/global vendors
(``alibaba`` / ``alibaba-cn``, ``minimax`` / ``minimax-cn``): the bare name is
the international endpoint, ``-cn`` is the mainland-China endpoint. It also
matches the ids models.dev already publishes for this vendor.
"""

from providers import register_provider
from providers.base import ProviderProfile

# Cheap, tool-capable and reachable on both sites — used for auxiliary work
# (context compression, vision, session titles) so those calls don't bill the
# user's flagship model. Mirrors the DeepInfra profile's choice.
_AUX_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# Curated fallbacks for the /model picker when the live /v1/models probe fails
# (no key yet, offline, rate-limited). Only agentic tool-capable models belong
# here — same rule as the other curated lists. The live catalog leads the picker.
#
# Every id below was verified against the live 中文站 catalog on 2026-09-09:
# present in /v1/models AND answering a real tool-call request. Do not add ids
# from models.dev alone — that catalog lags the platform. Qwen3-235B-A22B-
# Thinking-2507 was listed there but is disabled upstream (403 "Model
# disabled") and absent from the live catalog, so it is deliberately excluded.
_FALLBACK_MODELS = (
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-5.2",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
)

siliconflow = ProviderProfile(
    name="siliconflow",
    display_name="SiliconFlow",
    description="SiliconFlow (OpenAI-compatible aggregator for open-weight models)",
    signup_url="https://cloud.siliconflow.com/account/ak",
    env_vars=("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL"),
    base_url="https://api.siliconflow.com/v1",
    auth_type="api_key",
    default_aux_model=_AUX_MODEL,
    fallback_models=_FALLBACK_MODELS,
    # The /v1 chat endpoint hosts Qwen3-VL / GLM-V family models and accepts
    # image parts in tool-result messages like other OpenAI-compatible wires.
    supports_vision=True,
)

siliconflow_cn = ProviderProfile(
    name="siliconflow-cn",
    display_name="SiliconFlow (中文站)",
    description="硅基流动中文站 — 一个 Key 调用主流开源模型",
    signup_url="https://cloud.siliconflow.cn/account/ak",
    env_vars=("SILICONFLOW_CN_API_KEY", "SILICONFLOW_CN_BASE_URL"),
    base_url="https://api.siliconflow.cn/v1",
    auth_type="api_key",
    default_aux_model=_AUX_MODEL,
    fallback_models=_FALLBACK_MODELS,
    supports_vision=True,
)

register_provider(siliconflow)
register_provider(siliconflow_cn)
