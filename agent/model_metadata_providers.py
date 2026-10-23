"""Known provider hosts, including historical endpoints in saved routes."""

# Host substring -> provider. ".githubcopilot.com" covers api.enterprise./api.business.
# hosts; models.inference.ai.azure.com (GitHub Models free tier, ~8K per-request cap)
# is mapped so a targeted hint fires instead of the custom-endpoint path.
_URL_TO_PROVIDER: dict[str, str] = {
    "api.openai.com": "openai", "chatgpt.com": "openai", "api.anthropic.com": "anthropic",
    "api.z.ai": "zai", "open.bigmodel.cn": "zai",
    "api.moonshot.ai": "kimi-coding", "api.moonshot.cn": "kimi-coding-cn", "api.kimi.com": "kimi-coding",
    "api.stepfun.ai": "stepfun", "api.stepfun.com": "stepfun", "api.arcee.ai": "arcee", "api.minimax": "minimax",
    "dashscope.aliyuncs.com": "alibaba", "dashscope-intl.aliyuncs.com": "alibaba", "portal.qwen.ai": "qwen-oauth",
    # Historic Token Plan endpoints remain valid in saved configurations.
    "token-plan.ap-southeast-1.maas.aliyuncs.com": "alibaba-token-plan",
    "token-plan.cn-beijing.maas.aliyuncs.com": "alibaba-token-plan-cn",
    "openrouter.ai": "openrouter", "generativelanguage.googleapis.com": "gemini",
    "inference-api.nousresearch.com": "nous", "api.deepseek.com": "deepseek",
    "api.githubcopilot.com": "copilot", ".githubcopilot.com": "copilot", "models.github.ai": "copilot",
    "models.inference.ai.azure.com": "copilot",
    "api.fireworks.ai": "fireworks", "opencode.ai": "opencode-go", "api.x.ai": "xai",
    "integrate.api.nvidia.com": "nvidia", "api.xiaomimimo.com": "xiaomi", "xiaomimimo.com": "xiaomi",
    "api.gmi-serving.com": "gmi", "api.novita.ai": "novita",
    "tokenhub.tencentmaas.com": "tencent-tokenhub", "api.lkeap.cloud.tencent.com": "tencent-tokenplan",
    "ollama.com": "ollama-cloud",
}
