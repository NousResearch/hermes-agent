"""Auxiliary-task default blocks for ``DEFAULT_CONFIG`` (see ``config_defaults``).

Split out so ``config_defaults`` stays a pure-data leaf whose line budget may only go
down: this is the only code those defaults need, and it is used eighteen times by
``DEFAULT_CONFIG["auxiliary"]``.
"""


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
