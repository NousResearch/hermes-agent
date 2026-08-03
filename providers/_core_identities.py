"""Cycle-free identities for core model providers without plugin profiles.

Most built-in inference providers have a bundled ``model-provider`` plugin,
so their IDs intentionally remain governed by plugin activation.  These are
the exceptions: core runtime implementations that have no ProviderProfile at
all, plus the virtual MoA route.  Provider discovery imports this declarative
module before reading any external manifest, preventing an inactive external
plugin from claiming a core route or one of its pre-normalization aliases.
"""

PROFILELESS_CORE_PROVIDER_IDS = frozenset(
    {
        "lmstudio",
        "moa",
        "openai-api",
        "tencent-tokenhub",
        "xai-oauth",
    }
)

PROFILELESS_CORE_PROVIDER_ALIASES = {
    "grok-oauth": "xai-oauth",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "tencent": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "x-ai-oauth": "xai-oauth",
    "xai-grok-oauth": "xai-oauth",
}
