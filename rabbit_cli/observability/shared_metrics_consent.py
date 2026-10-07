"""The one shared-metrics consent answer, and the terminal's first-run offer.

Every surface reads and writes the same key in the profile's config.yaml:
``telemetry.shared_metrics.enabled`` (collect locally). Collection is local-only: nothing is
exported or uploaded. The question counts as answered once the key is written explicitly; the
shipped defaults are not an answer. Desktop paints the offer as a composer strip
(``consent-strip.tsx`` over the ``shared_metrics.*`` RPCs). The terminal asks at the end of
every ``rabbit setup`` flow and, for a profile that never answered, once before an interactive
``rabbit`` / ``rabbit --tui`` chat (which also covers the dashboard's Chat tab, a PTY-hosted
TUI). The messaging gateway never asks: a chat participant is not the install owner.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# (label, enabled): the Desktop strip's answers, in its order.
OFFER_CHOICES = (
    ("Collect locally only", True),
    ("No thanks", False),
)
_NO_THANKS = len(OFFER_CHOICES) - 1
DOCS_URL = "https://github.com/seven0070/Rabbit-/tree/main/website/docs/developer-guide/relay-shared-metrics"
_OFFER_DESCRIPTION = "\n".join((
    "Shared metrics are bounded counters: activity, outcomes, error classes, model routes,",
    "token totals, feature use and coarse machine facts. Never prompts, files, paths,",
    "setting values or error text. Collection stays on this machine; nothing is uploaded.",
    f"Details: {DOCS_URL}",
    "Change it any time: rabbit setup telemetry",
))


def _section(cfg: Any) -> dict:
    telemetry = cfg.get("telemetry") if isinstance(cfg, dict) else None
    section = telemetry.get("shared_metrics") if isinstance(telemetry, dict) else None
    return section if isinstance(section, dict) else {}


def consent_state(raw_cfg: Any) -> dict:
    """``{enabled, decided}`` from a RAW config mapping (no defaults merged).

    A stale ``send`` key from the upload era still counts as decided, and is dropped on the
    next save."""
    section = _section(raw_cfg)
    return {
        "enabled": section.get("enabled") is True,
        "decided": "enabled" in section or "send" in section,
    }


def consent_decided() -> bool:
    from rabbit_cli.config import read_raw_config

    return consent_state(read_raw_config())["decided"]


def _set_answer(target: dict, enabled: bool) -> None:
    telemetry = target.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = target["telemetry"] = {}
    section = telemetry.get("shared_metrics")
    if not isinstance(section, dict):
        section = telemetry["shared_metrics"] = {}
    section["enabled"] = enabled
    section.pop("send", None)


def save_consent(enabled: bool, config: dict | None = None) -> None:
    """Write the answer into the user's raw config.yaml now. ``save_config`` strips values equal
    to the shipped defaults, so a "No thanks" saved through it vanished and the profile was
    asked again on every surface. ``config`` (a caller's in-memory copy, e.g. the wizard's)
    gets the same answer so its later save agrees. Any stale ``send`` key is dropped."""
    from rabbit_cli.config import (
        _write_user_config, get_config_path, is_managed, managed_error, require_readable_config_before_write,
    )

    if is_managed():
        managed_error("save configuration")
        return
    config_path = get_config_path()
    user_config = require_readable_config_before_write(config_path)
    _set_answer(user_config, enabled)
    _write_user_config(config_path, user_config)
    if config is not None:
        _set_answer(config, enabled)


def offer_consent(config: dict | None = None) -> bool:
    """Ask once, with the Desktop strip's two answers; "No thanks" is the default so Enter never
    opts anyone in. Esc leaves the question open (asked again next time). True when answered."""
    from rabbit_cli.cli_output import print_info, print_success
    from rabbit_cli.curses_ui import curses_radiolist

    idx = curses_radiolist(
        "Help improve Rabbit?", [label for label, _ in OFFER_CHOICES], selected=_NO_THANKS, cancel_returns=-1,
        description=_OFFER_DESCRIPTION,
    )
    if idx < 0:
        print_info("Not answered; Rabbit will ask again. Decide any time with `rabbit setup telemetry`.")
        return False
    _, enabled = OFFER_CHOICES[idx]
    save_consent(enabled, config)
    outcome = "collected on this machine" if enabled else "off"
    print_success(f"Shared metrics {outcome}. Change it any time with `rabbit setup telemetry`.")
    return True


def offer_consent_if_undecided(config: dict | None = None) -> None:
    """The end-of-setup offer: every ``rabbit setup`` flow funnels through the setup-completed
    record, so this one call covers Quick, Full, Blank Slate, Portal and ``--quick``."""
    from rabbit_cli.config import is_managed
    from rabbit_cli.setup import is_noninteractive

    if is_managed() or is_noninteractive() or not sys.stdin.isatty() or consent_decided():
        return
    try:
        offer_consent(config)
    except (KeyboardInterrupt, EOFError):
        print()


def offer_consent_before_chat(args: Any) -> None:
    """One-time offer before an interactive chat on a profile that never answered. Skipped for
    anything without a person at a terminal (``-q``, piped or JSON output, spawned actions) and for
    a Desktop-spawned pane (the app shows its own strip)."""
    from rabbit_cli.config import is_managed
    from rabbit_cli.setup import is_noninteractive

    if any(getattr(args, name, None) for name in ("query", "query_file", "oneshot", "oneshot_exit")):
        return
    if getattr(args, "safe_mode", False) or getattr(args, "ignore_user_config", False):
        return
    if os.environ.get("RABBIT_DESKTOP") or is_noninteractive():
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or is_managed():
        return
    try:
        if consent_decided():
            return
        offer_consent()
    except (KeyboardInterrupt, EOFError):
        print()
    except Exception:
        logger.debug("shared-metrics offer skipped", exc_info=True)
