"""Configuration resolution for the LCM context engine plugin.

User-facing settings live in ``config.yaml`` under ``context.lcm``, never in
``.env`` (Hermes reserves the environment for secrets. See ``AGENTS.md``).

Two rules matter more than the values themselves:

* **Fail-open defaults.**  Every field has a default, and anything unparseable
  or nonsensical falls back to it.  A broken config block must never stop the
  engine from loading, and must never widen a recall bound.
* **Hard ceilings, not preferences.**  ``page_size`` / ``limit`` /
  ``body_chars`` are *clamped* to the module constants below.  The bounded
  recall contract ("never load an unbounded transcript") is enforced in code,
  so a hand-edited config cannot talk the engine out of it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Hard ceilings: the bounded-recall contract.  Config may lower these, never
# raise them: the values are the engine's promise to the host.
HARD_MAX_PAGE_SIZE = 50
HARD_MAX_SEARCH_RESULTS = 25
HARD_MAX_BODY_CHARS = 8_000
HARD_MAX_SUMMARY_CHARS = 6_000

# Defaults chosen to be useful on the first turn without any config block.
DEFAULT_PAGE_SIZE = 10
DEFAULT_SEARCH_RESULTS = 8
DEFAULT_BODY_CHARS = 2_000
DEFAULT_SUMMARY_CHARS = 4_000

DEFAULTS: Dict[str, Any] = {
    # Empty path -> ``<HERMES_HOME>/lcm/lcm.db`` (resolved by the engine).
    "db_path": "",
    "page_size": DEFAULT_PAGE_SIZE,
    "max_page_size": HARD_MAX_PAGE_SIZE,
    "max_search_results": DEFAULT_SEARCH_RESULTS,
    "body_chars": DEFAULT_BODY_CHARS,
    "summary_chars": DEFAULT_SUMMARY_CHARS,
    "protect_first_n": 3,
    "protect_last_n": 6,
    "threshold_percent": 0.75,
    "ingest_on_turn_complete": True,
}

_KNOWN_KEYS = frozenset(DEFAULTS)


def _as_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return default if default >= minimum else minimum
    return min(parsed, maximum)


def _as_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not (minimum <= parsed <= maximum):
        return default
    return parsed


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


class LCMConfig:
    """Resolved, clamped settings for one engine instance."""

    __slots__ = tuple(DEFAULTS) + ("extra",)

    def __init__(self, values: Optional[Dict[str, Any]] = None) -> None:
        raw: Dict[str, Any] = dict(DEFAULTS)
        extra: Dict[str, Any] = {}
        if isinstance(values, dict):
            for key, value in values.items():
                if key in _KNOWN_KEYS:
                    raw[key] = value
                else:
                    extra[key] = value

        self.db_path = str(raw.get("db_path") or "").strip()
        # ``max_page_size`` is itself clamped, and ``page_size`` never exceeds it.
        self.max_page_size = _as_int(
            raw.get("max_page_size"), HARD_MAX_PAGE_SIZE,
            minimum=1, maximum=HARD_MAX_PAGE_SIZE,
        )
        self.page_size = _as_int(
            raw.get("page_size"), min(DEFAULT_PAGE_SIZE, self.max_page_size),
            minimum=1, maximum=self.max_page_size,
        )
        self.max_search_results = _as_int(
            raw.get("max_search_results"), DEFAULT_SEARCH_RESULTS,
            minimum=1, maximum=HARD_MAX_SEARCH_RESULTS,
        )
        self.body_chars = _as_int(
            raw.get("body_chars"), DEFAULT_BODY_CHARS,
            minimum=200, maximum=HARD_MAX_BODY_CHARS,
        )
        self.summary_chars = _as_int(
            raw.get("summary_chars"), DEFAULT_SUMMARY_CHARS,
            minimum=200, maximum=HARD_MAX_SUMMARY_CHARS,
        )
        self.protect_first_n = _as_int(
            raw.get("protect_first_n"), DEFAULTS["protect_first_n"],
            minimum=0, maximum=50,
        )
        self.protect_last_n = _as_int(
            raw.get("protect_last_n"), DEFAULTS["protect_last_n"],
            minimum=0, maximum=200,
        )
        self.threshold_percent = _as_float(
            raw.get("threshold_percent"), DEFAULTS["threshold_percent"],
            minimum=0.05, maximum=0.99,
        )
        self.ingest_on_turn_complete = _as_bool(
            raw.get("ingest_on_turn_complete"), True
        )
        self.extra = extra

    def bounds(self) -> Dict[str, int]:
        """Return the effective (clamped) recall bounds, for diagnostics."""
        return {
            "page_size": self.page_size,
            "max_page_size": self.max_page_size,
            "max_search_results": self.max_search_results,
            "body_chars": self.body_chars,
            "summary_chars": self.summary_chars,
        }

    def as_dict(self) -> Dict[str, Any]:
        data = {key: getattr(self, key) for key in DEFAULTS}
        data["extra"] = dict(self.extra)
        return data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"LCMConfig({self.as_dict()!r})"


def load_lcm_config(config: Optional[Dict[str, Any]] = None) -> LCMConfig:
    """Resolve ``context.lcm`` from the live config, falling back to defaults.

    ``config`` may be passed explicitly (tests). Otherwise the loaded Hermes
    config is used, and any failure to read it yields pure defaults.
    """
    block: Any = None
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = None
    if isinstance(config, dict):
        context = config.get("context")
        if isinstance(context, dict):
            block = context.get("lcm")
    return LCMConfig(block if isinstance(block, dict) else {})
