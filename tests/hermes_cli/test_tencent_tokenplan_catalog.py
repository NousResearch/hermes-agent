"""通用 Token Plan 的模型目录：文档 130060 列出的 Model ID 必须可选，服务端已下线的不得列。"""

# 文档 https://cloud.tencent.com/document/product/1823/130060 「通用 Token Plan → 可用模型」
DOC_MODEL_IDS = [
    "tc-code-latest",
    "deepseek-v4-flash-202605",
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-flash",
    "deepseek-v4-pro-202606",
    "deepseek/deepseek-v4-pro-0813",
    "deepseek/deepseek-v4-pro",
    "minimax-m2.7",
    "minimax-m-2-7",
    "minimax-m3",
    "minimax-m-3-0",
    "glm-5",
    "glm-5-0",
    "glm-5.1",
    "glm-5-1",
    "glm-5.2",
    "glm-5-2",
    "glm-5.3",
    "glm-5-3",
    "glm-5.3-flash",
    "hy4-preview",
    "kimi-k2.7-code",
    "kimi-k3",
]

# Hy Token Plan 专属，不属于通用套餐；实测 /plan/v3 返回 20098 token plan quota exhausted
HY_PLAN_ONLY = ["hy3", "hy3-preview", "hy3-202608"]


def test_tencent_tokenplan_static_catalog_covers_doc_models():
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    catalog = _PROVIDER_MODELS["tencent-tokenplan"]
    missing = [m for m in DOC_MODEL_IDS if m not in catalog]
    assert not missing, f"missing from tencent-tokenplan catalog: {missing}"


def test_tencent_tokenplan_excludes_hy_plan_models():
    """hy3* 是 Hy Token Plan 模型，出现在通用套餐目录里会误导切换。"""
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    catalog = _PROVIDER_MODELS["tencent-tokenplan"]
    leaked = [m for m in HY_PLAN_ONLY if m in catalog]
    assert not leaked, f"Hy-plan-only models leaked into tencent-tokenplan: {leaked}"


def test_tencent_tokenplan_excludes_retired_kimi_k25():
    """kimi-k2.5 / kimi-k-2-5 服务端稳定返回 20057 model engine error（OpenAI 与 Anthropic 端点皆是）。"""
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    catalog = _PROVIDER_MODELS["tencent-tokenplan"]
    present = [m for m in ("kimi-k2.5", "kimi-k-2-5") if m in catalog]
    assert not present, f"broken Kimi K2.5 ids in catalog: {present}"


def test_tencent_tokenhub_keeps_hy_models():
    """hy3/hy3-preview 属于 TokenHub/Hy 侧目录，不能因为它们从通用套餐挪走而丢失。"""
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    hub = _PROVIDER_MODELS["tencent-tokenhub"]
    for m in ("hy4-preview", "hy3", "hy3-preview"):
        assert m in hub, f"tencent-tokenhub lost {m}"
