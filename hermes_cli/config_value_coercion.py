"""Value coercion for ``hermes config set`` and dynamic platform string enums."""

from typing import Any

import hermes_yaml as yaml


def _is_platform_reply_to_mode_key(key: str) -> bool:
    """Whether *key* names the reply-mode string enum on a dynamic platform.

    This is intentionally exact rather than a broad suffix rule: other dynamic
    adapter fields retain their existing scalar coercion until they define their
    own string-enum contract.
    """
    from hermes_cli.config import _split_key_path

    parts = _split_key_path(key)
    return len(parts) == 3 and parts[0] == "platforms" and parts[2] == "reply_to_mode"


def _coerce_config_set_value(key: str, value: str) -> Any:
    """Auto-coerce a ``hermes config set`` string to bool/None/int/float/list/dict.
    String-typed settings (per ``DEFAULT_CONFIG``) are preserved verbatim so enum members such as
    ``approvals.mode="off"`` never become booleans. List/mapping literals are parsed so
    isinstance-gated readers see real structures; the trigger is conservative.
    Bare ``model`` is the exception: its string default is the model-id shorthand, so a structured
    literal there is parsed for the section guard to gate instead of riding into model.default
    as a bogus id (#131435)."""
    from hermes_cli.config import (
        _SCALAR_WORDS, _coerce_float, _coerce_int, _default_value_for_key,
        _exit_invalid, _looks_structured_value,
    )

    if _is_platform_reply_to_mode_key(key) or (
            isinstance(_default_value_for_key(key), str) and not (
                key == "model" and _looks_structured_value(value))):
        return value
    stripped = value.strip()
    lower = stripped.lower()
    if lower in _SCALAR_WORDS:
        return _SCALAR_WORDS[lower]
    for coerce in (_coerce_int, _coerce_float):
        coerced = coerce(stripped)
        if coerced is not None:
            return coerced
    if not _looks_structured_value(value):
        return value
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        # Storing the text as a string here used to be a warning; every isinstance-gated reader
        # then ignored the value while `config get` echoed it back (#114471). Refuse instead.
        detail = str(getattr(exc, "problem", None) or exc).splitlines()[0]
        _exit_invalid(
            f"✗ Value for '{key}' looks like a list/mapping but is not valid YAML/JSON "
            f"({detail}) — nothing was written.\n"
            "  Fix the literal, or quote it (e.g. \"'[text'\") to store a plain string.")
    if isinstance(parsed, (list, dict)):
        return parsed
    # A quoted literal ("'[text'") parses to a scalar: that is the deliberate way to store one.
    return value
