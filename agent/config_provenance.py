"""Tell an operator's deliberate value apart from a shipped default.

``load_config()`` deep-merges ``DEFAULT_CONFIG`` under the user's file, so a key
with a shipped default is **always present** in the merged mapping. Any code that
infers "the operator set this" from ``cfg.get(k) is not None`` is therefore
permanently wrong for every defaulted key — it reads a default as an intentional
override.

This is not hypothetical. Worked example:

* ``compression.context_timeout_seconds`` ships ``120`` in ``config_defaults.py``.
* ``resolve_context_compression_timeouts()`` set ``explicit_idle=True`` because the
  key was present.
* ``reconcile_idle_timeout()`` honours ``explicit=True`` verbatim **by design**
  (an operator who names a number means it), so it returned 120 unchanged.
* The outer no-progress guard therefore still fired at 120 s, BEFORE the 300 s
  inner auxiliary deadline — which is precisely the outer-guard defect this module exists to
  prevent. ``call_llm`` never raised, so the 9 configured ``fallback_providers``
  stayed structurally unreachable on a stalled summariser.

The fix landed correct and was **inert in production**, because its own caller
could not distinguish default-merged from user-set.

The sibling that got it right is ``agent/hygiene_timeout.py``: its key ships
``None``, so presence genuinely means user-set. That works, but it constrains the
schema (you cannot document a real default) and it silently breaks the moment
someone gives the key a non-``None`` default. This module removes the constraint:
compare against ``DEFAULT_CONFIG`` instead of relying on a sentinel.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["default_config_value", "is_operator_set", "MISSING"]


class _Missing:
    """Sentinel for 'this key has no shipped default at all'."""

    _singleton: Optional["_Missing"] = None

    def __new__(cls) -> "_Missing":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


def default_config_value(*path: str) -> Any:
    """Return the value ``DEFAULT_CONFIG`` ships at ``path``, or :data:`MISSING`.

    Never raises: a missing/undentifiable defaults module means "no shipped
    default", which makes :func:`is_operator_set` fall back to presence — the
    historical behaviour, so this can only ever be an improvement.
    """
    try:
        from hermes_cli.config_defaults import DEFAULT_CONFIG
    except Exception:  # pragma: no cover - defensive
        return MISSING

    node: Any = DEFAULT_CONFIG
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return MISSING
        node = node[key]
    return node


def is_operator_set(cfg: Any, key: str, *default_path: str) -> bool:
    """True when ``cfg[key]`` is a deliberate operator value, not a shipped default.

    ``default_path`` locates the same key inside ``DEFAULT_CONFIG`` (e.g.
    ``("compression", "context_timeout_seconds")``). When the shipped default and
    the observed value are equal, the operator has NOT expressed an intent — even
    if they happened to type the same number, in which case honouring the derived
    value is still correct (identical value, better floor).

    Returns False for an absent key. Falls back to presence when the key has no
    shipped default, which preserves the historical contract for keys that are
    genuinely absent from ``DEFAULT_CONFIG``.
    """
    if not isinstance(cfg, Mapping) or key not in cfg:
        return False

    shipped = default_config_value(*(default_path or (key,)))
    if shipped is MISSING:
        # No default to compare against: presence IS intent, as before.
        return True

    observed = cfg.get(key)
    if observed is None and shipped is None:
        return False

    # bool is an int subclass; compare types first so True != 1 sneaking through
    # a numeric compare can't mark a default as operator-set.
    if isinstance(observed, bool) != isinstance(shipped, bool):
        return True

    try:
        return observed != shipped
    except Exception:  # pragma: no cover - exotic types
        return True
