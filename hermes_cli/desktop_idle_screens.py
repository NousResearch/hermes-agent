"""Bot Desktop idle stop for every profile a Desktop-owned ``hermes serve`` backend serves (#133660).

The serve-side lease watcher (``tui_gateway/methods_display_watch.py``) starts only after a client's
first ``display.*`` call and visits only the launch home plus profiles a client has addressed since
this process started. On a host whose cron runs in the Desktop backend (no messaging gateway), a
screen that ``bot_desktop.auto_start`` brought up for a cron turn of a profile nobody opened in
Desktop therefore never stopped. This ticker starts with the backend and walks the same served set
as the Desktop cron ticker. A profile a live gateway owns is skipped: the gateway's housekeeping
stops its screens, so each profile has one owner. ``runtime.stop()`` is lock-guarded, so the lease
watcher may also run.
"""

from __future__ import annotations

import logging
import threading

_log = logging.getLogger(__name__)


def stop_idle_screens() -> None:
    from hermes_cli.profiles import profiles_to_serve
    from hermes_cli.web_server import _gateway_owns_cron
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.bot_desktop import runtime

    for name, home in profiles_to_serve(multiplex=True):
        # One boundary per profile: an unreadable profile must not stop the walk for the rest.
        try:
            if _gateway_owns_cron(name, home):
                continue
            token = set_hermes_home_override(str(home))
            try:
                runtime.stop_if_idle()
            finally:
                reset_hermes_home_override(token)
        except Exception:  # health: allow BLE001 -- per-profile boundary, as gateway _for_each_served_profile
            _log.debug("Bot Desktop idle stop skipped for profile %s", name, exc_info=True)


def run_idle_screen_ticker(stop_event: threading.Event, interval: int = 60) -> None:
    while not stop_event.wait(interval):
        stop_idle_screens()
