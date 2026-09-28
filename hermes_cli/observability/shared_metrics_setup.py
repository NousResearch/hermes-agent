"""hermes.provider_setup.count: where connecting a model provider starts, succeeds, fails or is walked away from.

A flow is ``started`` once the user has picked a provider. Its end is recorded by the surface that
ran it (``completed`` / ``failed`` + a closed failure class). A flow nobody finished cannot record
itself, so ``started`` also drops a small marker under the owning profile's store dir (the
process-exit pattern): the finisher claims it by rename, and a marker whose process is gone (or that
has been pending longer than any real flow takes) is reported ``abandoned`` by the next setup start
or Hermes start in that profile. Whoever claims the marker records, so each flow ends exactly once.

Only the catalog provider name leaves (custom endpoints read ``custom``); never a key, token, base
URL or error text. Rows are saved synchronously: a setup killed right after ``started`` must still
have left its row, and none of this runs on a hot path.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger(__name__)

MARKER_DIRNAME = "provider_setup_markers"
# Device codes expire in ~15 minutes; a flow still pending an hour later was left open and forgotten.
STALE_AFTER_S = 3600
# Picker rows that are not a provider (menus, removal) never start a setup flow.
_NOT_A_PROVIDER = frozenset({"aux-config", "cancel", "reasoning", "remove-custom"})
_AUTH_ERROR_TYPES = frozenset({"AuthError", "SignInCopyError"})
_OAUTH_SESSION_KEY = "_metrics_setup_flow"
# OAuth session status (+ reason) at its end -> (event, failure_class).
_OAUTH_ENDINGS = {
    "approved": ("completed", "none"),
    "cancelled": ("failed", "cancelled"),
    "denied": ("failed", "auth"),
    "expired": ("abandoned", "none"),
    "error": ("failed", "other"),
}
_OAUTH_REASONS = {
    "user_declined": ("failed", "cancelled"),
    "superseded": ("failed", "cancelled"),
    "timeout": ("abandoned", "none"),
    "account_retired": ("failed", "auth"),
}


@dataclass
class SetupFlow:
    home: str
    surface: str
    provider: str  # already the catalog metric name
    marker: Path


def markers_dir(home: Path) -> Path:
    return home / "telemetry" / "shared_metrics" / MARKER_DIRNAME


@contextlib.contextmanager
def _bound(hermes_home: Any) -> Iterator[None]:
    token = None
    if hermes_home:
        from hermes_constants import set_hermes_home_override

        token = set_hermes_home_override(str(hermes_home))
    try:
        yield
    finally:
        if token is not None:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)


def collection_enabled(hermes_home: Any = None) -> bool:
    """The owning profile's shared-metrics gate (cheap; lets async callers skip the thread hop)."""
    try:
        with _bound(hermes_home):
            from .relay_shared_metrics import enabled

            return bool(enabled())
    except Exception:
        return False


def provider_setup_fields(*, surface: Any, provider: Any, event: Any, failure_class: Any = "none") -> dict[str, str] | None:
    from . import shared_metrics_contract as contract
    from .shared_metrics_catalog import provider_metric_name

    if surface not in contract.PROVIDER_SETUP_SURFACES or event not in contract.PROVIDER_SETUP_EVENTS:
        return None
    failure = "none"
    if event == "failed":
        failure = failure_class if failure_class in contract.PROVIDER_SETUP_FAILURE_CLASSES - {"none"} else "other"
    return {"event": event, "failure_class": failure, "provider": provider_metric_name(provider), "surface": surface}


def setup_failure_class(exc: BaseException) -> str:
    """Closed class for an exception that ended a setup flow; inert (type checks only)."""
    from hermes_cli.auth_error_copy import is_cancelled, is_network_error

    if is_cancelled(exc):
        return "cancelled"
    if is_network_error(exc):
        return "network"
    if {cls.__name__ for cls in type(exc).__mro__} & _AUTH_ERROR_TYPES:
        return "auth"
    return "other"


def _save_row(fields: dict[str, str] | None) -> bool:
    """Save one row and wait for the store; a row the contract cannot build is settled as is."""
    if fields is None:
        return True
    from .shared_metrics_contract import PROVIDER_SETUP_MARK
    from .shared_metrics_events import emit_saved

    return emit_saved([(PROVIDER_SETUP_MARK, fields)]) == 1


def begin_provider_setup(surface: str, provider: Any, *, hermes_home: Any = None) -> SetupFlow | None:
    """Record ``started`` and drop the pending marker; None (no files, no rows) when collection is off."""
    try:
        with _bound(hermes_home):
            from .relay_shared_metrics import enabled

            if not provider or provider in _NOT_A_PROVIDER or not enabled():
                return None
            fields = provider_setup_fields(surface=surface, provider=provider, event="started")
            if fields is None:
                return None
            from gateway.status import get_process_start_time
            from hermes_constants import get_hermes_home

            home = get_hermes_home()
            report_abandoned_setups(home)
            directory = markers_dir(home)
            directory.mkdir(parents=True, exist_ok=True)
            pid = os.getpid()
            marker = directory / f"{surface}-{pid}-{time.time_ns()}.json"
            record = {
                "pid": pid, "start_time": get_process_start_time(pid), "started_at": time.time(),
                "surface": surface, "provider": fields["provider"],
            }
            tmp = marker.with_name(f".{marker.name}.tmp")
            tmp.write_text(json.dumps(record), encoding="utf-8")
            os.replace(tmp, marker)
            _save_row(fields)
            return SetupFlow(home=str(home), surface=surface, provider=fields["provider"], marker=marker)
    except Exception:
        logger.debug("Provider setup start not recorded", exc_info=True)
        return None


def finish_provider_setup(flow: SetupFlow | None, event: str, failure_class: str = "none") -> None:
    """Record the flow's end once: only if its marker is still unclaimed (never raises)."""
    if flow is None:
        return
    try:
        from .shared_metrics_process import settle_claim

        claimed = flow.marker.with_name(f"{flow.marker.name}.{os.getpid()}.reporting")
        try:
            os.replace(flow.marker, claimed)
        except OSError:  # already reported (abandoned) by another start
            return
        with _bound(flow.home):
            fields = provider_setup_fields(
                surface=flow.surface, provider=flow.provider, event=event, failure_class=failure_class,
            )
            settle_claim(claimed, flow.marker, _save_row(fields))
    except Exception:
        logger.debug("Provider setup end not recorded", exc_info=True)


def _reportable(record: Any) -> bool:
    from gateway.status import runtime_status_pid_is_live

    from .shared_metrics_contract import _non_negative_number

    if not isinstance(record, dict):
        return True
    started = _non_negative_number(record.get("started_at"))
    if started is None or time.time() - started > STALE_AFTER_S:
        return True
    return not runtime_status_pid_is_live(record)


def report_abandoned_setups(home: Path) -> None:
    """Report every pending marker whose flow can no longer finish as ``abandoned``.

    Callers bind ``home`` (the scan runs at a setup start or on the process-exit reporter thread).
    """
    from .shared_metrics_process import _claimer_alive, settle_claim

    directory = markers_dir(home)
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or ".json" not in path.name:
            continue
        if path.name.endswith(".reporting") and _claimer_alive(path):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            record = None
        if not path.name.endswith(".reporting") and not _reportable(record):
            continue
        original = path.with_name(path.name.split(".json")[0] + ".json")
        claimed = original.with_name(f"{original.name}.{os.getpid()}.reporting")
        try:
            os.replace(path, claimed)
        except OSError:  # a concurrent reporter or the finisher won
            continue
        fields = provider_setup_fields(
            surface=record.get("surface"), provider=record.get("provider"), event="abandoned",
        ) if isinstance(record, dict) else None
        settle_claim(claimed, original, _save_row(fields))


def record_provider_setup_done(surface: str, provider: Any, *, hermes_home: Any = None, background: bool = False) -> None:
    """A setup that starts and lands in one action (an API key saved from a form): both rows at once.
    ``background`` keeps a request handler from waiting on a cold metrics runtime; the owning home is
    captured here, since a thread does not inherit the profile binding."""
    try:
        with _bound(hermes_home):
            from .relay_shared_metrics import enabled

            if not provider or not enabled():
                return
            from hermes_constants import get_hermes_home

            home = str(get_hermes_home())

        def run() -> None:
            finish_provider_setup(begin_provider_setup(surface, provider, hermes_home=home), "completed")

        if background:
            threading.Thread(target=run, name="hermes-setup-metrics", daemon=True).start()
        else:
            run()
    except Exception:
        logger.debug("Provider setup not recorded", exc_info=True)


def provider_for_api_key_env(env_var: Any) -> str | None:
    """The shipped provider whose API key lives in ``env_var``; None for any other variable."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    if env_var == "OPENROUTER_API_KEY":  # the aggregator is not a registry entry
        return "openrouter"
    return next((slug for slug, pconfig in PROVIDER_REGISTRY.items()
                 if env_var in (getattr(pconfig, "api_key_env_vars", None) or ())), None)


def record_api_key_saved(env_var: Any, surface: str) -> None:
    """Count a provider API key saved from a settings form (other variables are not setups)."""
    try:
        from .relay_shared_metrics import enabled

        if enabled() and (provider := provider_for_api_key_env(env_var)):
            record_provider_setup_done(surface, provider, background=True)
    except Exception:
        logger.debug("API key setup not recorded", exc_info=True)


# ---- CLI flows (`hermes setup`, `hermes model`, first-run chat) ------------------------------

_cli = threading.local()


def note_provider_setup_saved() -> None:
    """A CLI flow persisted the provider/model choice (called by the shared save helpers)."""
    hints = getattr(_cli, "hints", None)
    if hints is not None:
        hints["saved"] = True


def note_provider_setup_failure(failure_class: str) -> None:
    """A CLI flow reported a failure it then returned from instead of raising (first one wins)."""
    hints = getattr(_cli, "hints", None)
    if hints is not None and hints.get("failure") is None:
        hints["failure"] = failure_class


def note_sign_in_failure(exc: BaseException) -> None:
    with contextlib.suppress(Exception):
        note_provider_setup_failure(setup_failure_class(exc))


def _model_route() -> tuple[Any, ...]:
    from hermes_cli.config import load_config

    model = load_config().get("model")
    if not isinstance(model, dict):
        return (model,)
    return tuple(model.get(key) for key in ("provider", "default", "base_url"))


@contextlib.contextmanager
def provider_setup_surface(surface: str) -> Iterator[None]:
    """Name the surface whose provider picker runs inside (``hermes model``, the setup wizard, the
    first-run chat prompt). The picker is shared, so only these entry points know which one it is."""
    previous = getattr(_cli, "surface", None)
    _cli.surface = surface
    try:
        yield
    finally:
        _cli.surface = previous


@contextlib.contextmanager
def cli_provider_setup(provider: Any) -> Iterator[None]:
    """Track one CLI provider flow under the entry point's surface. Without one (fallback picking,
    tests driving the picker directly) nothing is tracked.

    Completed when the flow saved a choice or changed the model route; a flow that returns without
    either failed with the class it reported, else the user backed out (``cancelled``).
    """
    surface = getattr(_cli, "surface", None)
    flow = begin_provider_setup(surface, provider) if surface else None
    if flow is None:
        yield
        return
    before = None
    with contextlib.suppress(Exception):
        before = _model_route()
    hints: dict[str, Any] = {"saved": False, "failure": None}
    _cli.hints = hints
    try:
        yield
    except BaseException as exc:
        _cli.hints = None
        finish_provider_setup(flow, "failed", setup_failure_class(exc))
        raise
    _cli.hints = None
    landed = hints["saved"]
    if not landed:
        with contextlib.suppress(Exception):
            landed = before is not None and _model_route() != before
    if landed:
        finish_provider_setup(flow, "completed")
    else:
        finish_provider_setup(flow, "failed", hints["failure"] or "cancelled")


# ---- dashboard / Desktop OAuth sessions ------------------------------------------------------

def web_setup_surface() -> str:
    from hermes_cli.process_identity import is_desktop_owned_backend

    return "desktop" if is_desktop_owned_backend() else "dashboard"


def begin_oauth_setup(provider: str, hermes_home: Any) -> SetupFlow | None:
    try:
        return begin_provider_setup(web_setup_surface(), provider, hermes_home=hermes_home)
    except Exception:
        logger.debug("OAuth setup start not recorded", exc_info=True)
        return None


def attach_oauth_setup(sess: dict[str, Any] | None, flow: SetupFlow | None) -> None:
    if flow is not None and isinstance(sess, dict):
        sess[_OAUTH_SESSION_KEY] = flow


def settle_oauth_setup(sess: dict[str, Any] | None, *, abandoned: bool = False) -> None:
    """End an OAuth session's flow from its status (no-op while pending, or when already ended)."""
    try:
        if not isinstance(sess, dict) or sess.get(_OAUTH_SESSION_KEY) is None:
            return
        status = "cancelled" if sess.get("cancelled") else str(sess.get("status") or "")
        ending = _OAUTH_REASONS.get(str(sess.get("reason") or "")) if status in {"denied", "error"} else None
        ending = ending or _OAUTH_ENDINGS.get(status)
        if ending is None:
            if not abandoned:
                return
            ending = ("abandoned", "none")
        flow = sess.pop(_OAUTH_SESSION_KEY, None)
        finish_provider_setup(flow, *ending)
    except Exception:
        logger.debug("OAuth setup end not recorded", exc_info=True)
