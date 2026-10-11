"""Small reusable builders for :mod:`hermes_cli.config_defaults`."""


def auxiliary_model_defaults(timeout, *, reasoning_effort=True, **extra):
    """Build the standard auxiliary-task model block.

    ``reasoning_effort=False`` omits that key because MoA slots configure depth
    separately; ``extra`` keys are appended after the standard ones.
    """
    defaults = {
        "provider": "auto",
        "model": "",
        "base_url": "",
        "api_key": "",
        "timeout": timeout,
        "extra_body": {},
    }
    if reasoning_effort:
        defaults["reasoning_effort"] = ""
    defaults.update(extra)
    return defaults
