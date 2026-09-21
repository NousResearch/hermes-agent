"""Shared-metrics consent: one global answer, per-profile overrides, one writer.

Consent for anonymous usage metrics is a decision about the PERSON, not about a profile, so
the first answer given on any surface (onboarding checkbox, CLI/desktop prompt, wizard,
``hermes tools``, Settings) is recorded once at the Hermes root
(``<root>/telemetry-consent.json``) and inherited by every profile that has not set
``telemetry.shared_metrics.enabled`` itself. That is a deliberate, product-approved exception
to "profiles are independent islands" and applies ONLY to this consent. A profile's own
explicit ``enabled``/``send`` keys always win over the global answer, so any profile can opt
out (or in) locally afterwards.

Every surface writes through ``apply_shared_metrics_choice``: it sets the profile keys
explicitly, records the global answer the first time, and reconciles the send-consent window
at the moment of the decision — the same writer the relay and the sender use, so no two
callers can disagree about when consent was observed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

SHARED_METRICS_PATH = ("telemetry", "shared_metrics")
GLOBAL_CONSENT_FILENAME = "telemetry-consent.json"


class SharedMetricsChoice(NamedTuple):
    enabled: bool
    send: bool


class SharedMetricsState(NamedTuple):
    enabled: bool
    send: bool
    #: Some surface has recorded an answer for this user (the global file exists). Off +
    #: undecided is the one state in which a surface may ask, once.
    decided: bool
    #: Where ``enabled``/``send`` came from: this profile's explicit keys, the global answer,
    #: or the built-in default (off).
    source: str  # "profile" | "global" | "default"


# ── global answer ────────────────────────────────────────────────────────────

def global_consent_path() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / GLOBAL_CONSENT_FILENAME


# (path, mtime_ns, size) -> answer. The relay gate reads this 2-3x per agent turn.
_global_cache: tuple[tuple[str, int, int], Optional[bool]] | None = None


def read_global_consent() -> Optional[bool]:
    """The user's recorded answer (True = share), or None when nobody has ever answered."""
    global _global_cache
    path = global_consent_path()
    try:
        stat = path.stat()
    except OSError:
        _global_cache = None
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _global_cache is not None and _global_cache[0] == key:
        return _global_cache[1]
    answer: Optional[bool] = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("share"), bool):
            answer = data["share"]
    except (OSError, ValueError):
        logger.debug("Unreadable shared-metrics consent file %s", path, exc_info=True)
    _global_cache = (key, answer)
    return answer


def write_global_consent(share: bool) -> None:
    from utils import atomic_json_write

    path = global_consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(
        path,
        {"share": share, "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        mode=0o600,
    )


# ── effective state ──────────────────────────────────────────────────────────

def _shared_node(config: dict) -> dict:
    node: object = config
    for key in SHARED_METRICS_PATH:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def shared_metrics_state(config: dict) -> SharedMetricsState:
    """Effective consent for the profile whose RAW config this is.

    Profile keys are tri-state: an explicit boolean ``enabled`` wins; absent means inherit the
    global answer; no global answer means off. ``send`` is only ever True together with
    ``enabled``.
    """
    shared = _shared_node(config)
    explicit = shared.get("enabled")
    global_answer = read_global_consent()
    if isinstance(explicit, bool):
        enabled = explicit
        send = enabled and shared.get("send") is True
        source = "profile"
    elif global_answer is not None:
        enabled = send = global_answer
        source = "global"
    else:
        enabled = send = False
        source = "default"
    return SharedMetricsState(enabled=enabled, send=send, decided=global_answer is not None, source=source)


def consent_prompt_pending(config: dict) -> bool:
    """True when a surface should ask: collection is off for this profile and nobody has answered."""
    state = shared_metrics_state(config)
    return not state.enabled and not state.decided


# ── the writer ───────────────────────────────────────────────────────────────

def apply_shared_metrics_choice(config: dict, *, enabled: bool, send: bool) -> SharedMetricsChoice:
    """Record a decision: explicit profile keys, the global answer (first time only), consent window.

    ``send`` is forced off when ``enabled`` is off: ``send: true`` with nothing collected would
    log an error every run and never send. The consent window is reconciled unconditionally —
    the send key may already be false while a window is still open, and it must close.
    """
    shared: dict = config
    for key in SHARED_METRICS_PATH:
        child = shared.get(key)
        if not isinstance(child, dict):
            child = shared[key] = {}
        shared = child
    choice = SharedMetricsChoice(enabled=enabled, send=enabled and send)
    shared["enabled"] = choice.enabled
    shared["send"] = choice.send
    if read_global_consent() is None:
        # The first explicit answer anywhere becomes the default every other profile inherits.
        try:
            write_global_consent(choice.send)
        except OSError:
            logger.warning("Unable to record the global shared-metrics answer", exc_info=True)
    record_send_consent_change(enabled=choice.send)
    return choice


def maybe_prompt_for_consent_cli() -> bool:
    """Ask once, on an interactive terminal, when nobody has answered yet.

    The interactive CLI calls this right before the REPL takes the terminal. Returns True when
    an answer was recorded. Non-interactive runs, managed installs (config is read-only there)
    and decided users never see the question.
    """
    import sys

    from hermes_cli.config import is_managed, read_raw_config, save_config

    if not (sys.stdin.isatty() and sys.stdout.isatty()) or is_managed():
        return False
    config = read_raw_config()
    if not consent_prompt_pending(config):
        return False

    from hermes_cli.cli_output import print_info
    from hermes_cli.setup import prompt_yes_no

    print_info(
        "Hermes can share anonymous usage metrics with Nous Research — bounded counters only, "
        "never prompts, files or personal data. Change any time with `hermes setup telemetry`."
    )
    share = prompt_yes_no("Share anonymous usage metrics with Nous Research?", default=True)
    apply_shared_metrics_choice(config, enabled=share, send=share)
    save_config(config)
    return True


def record_send_consent_change(*, enabled: bool) -> None:
    """Reconcile consent windows now, through the single consent writer."""
    try:
        from hermes_cli.observability.shared_metrics import SharedMetricsStore
        from hermes_cli.observability.shared_metrics_sender import reconcile_send_consent
        from hermes_cli.sqlite_util import write_txn

        with SharedMetricsStore()._connection() as connection, write_txn(connection):
            reconcile_send_consent(connection, enabled)
    except Exception:
        # Never block the caller on telemetry bookkeeping; the relay reconciles on the next hook.
        logger.debug("Unable to record shared-metrics consent change", exc_info=True)
