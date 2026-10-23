# GENERATED file - do not hand-edit. Source: the measured Token Plan catalogue
# (public copy: https://github.com/oliver-mee/alibaba-token-plan-wiki, data/models.json).
# Regenerate upstream and copy this file over to update.
#
# Personal and Team have DIFFERENT catalogues and the sk-sp- key prefix is identical
# for both, so the tier cannot be detected from the key.
#
# PREFER live /models over these lists. Verified 2026-07-21 with active keys for both
# tiers: /models is tier-aware, so live discovery gives each user the listed
# entitlement without needing to detect the tier. Known callable-but-unlisted
# IDs are retained separately because exact-ID inference can be available even
# when Alibaba omits an ID from /models.
# These static lists are the OFFLINE FALLBACK for when the probe fails. PERSONAL is the
# safer default there, since nobody is offered something outside its tier catalogue.
#
# Always handle 'Access to model denied' at call time regardless: a LAPSED subscription
# still lists its full catalogue on /models and then denies every completion.

# PERSONAL: 12 chat models
PERSONAL_MODELS = (
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-flash",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "deepseek-v4-flash-0731",
    "deepseek-v4.1-flash",
    "deepseek-v4.1-flash-1017",
    "glm-5.2",
    "glm-5.3",
)

# TEAM: 19 chat models
TEAM_MODELS = (
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "deepseek-v4-flash-0731",
    "deepseek-v4.1-flash",
    "deepseek-v4.1-flash-1017",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "kimi-k2.5",
    "glm-5.2",
    "glm-5.3",
    "glm-5.1",
    "glm-5",
    "MiniMax-M2.5",
)

# Unlisted-but-servable ids: answer HTTP 200 when called by exact id yet are
# deliberately ABSENT from the /models discovery listing (measured gateway
# behaviour; the /models list is the entitlement surface, and Alibaba's
# discovery omits these ids regardless). The plugin must keep offering them
# at call time even though live discovery never returns them.
UNLISTED_MODELS = (
)

DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_AUX_MODEL = "qwen3.6-flash"
