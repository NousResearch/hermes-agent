"""Pure-data helpers for auxiliary model default blocks."""


def aux_block(timeout, *, reasoning_effort=True, **extra):
    """Standard auxiliary-task block for DEFAULT_CONFIG['auxiliary'].

    reasoning_effort=False omits the key (MoA configures depth per slot).
    Extra keys follow the standard ones.
    """
    d = {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": timeout, "extra_body": {}}
    if reasoning_effort:
        d["reasoning_effort"] = ""
    d.update(extra)
    return d
