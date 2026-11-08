"""Pure-data factory for auxiliary model configuration defaults."""


def _aux(timeout, *, reasoning_effort=True, **extra):
    """Standard auxiliary-task model block (see DEFAULT_CONFIG["auxiliary"]).

    reasoning_effort=False omits that key (MoA blocks configure depth per slot);
    ``extra`` keys are appended after the standard ones.
    """
    d = {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": timeout, "extra_body": {}}
    if reasoning_effort:
        d["reasoning_effort"] = ""
    d.update(extra)
    return d


def title_generation_defaults():
    """Auxiliary title_generation block (moved out of config_defaults to keep that facade shrinking)."""
    return {
        "enabled": True,
        "model_upgrade_enabled": True,  # False = keep the instant derived title, never call a model
        # session_search no longer uses an aux LLM (#27590); leftover auxiliary.session_search is ignored.
        "provider": "auto",
        "model": "",
        "prefer_fast_model": False,
        "base_url": "",
        "api_key": "",
        "timeout": 30,
        "extra_body": {},
        "reasoning_effort": "",
        "language": "",
    }
