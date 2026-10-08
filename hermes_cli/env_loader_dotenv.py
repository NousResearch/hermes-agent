"""Dotenv layer parsing, ``${VAR}`` resolution and publication into ``os.environ``, shared by the
startup loader (``hermes_cli.env_loader.load_hermes_dotenv``) and the config-backend bootstrap that
runs before it. Secret sources and startup orchestration stay in ``hermes_cli.env_loader``."""
from __future__ import annotations

import codecs
import io
import itertools
import os
import threading
from pathlib import Path


# What THIS process has published from dotenv files, per variable: (value before our first publish, or
# None if absent; last value we published; load pass that published it). Reloads run per gateway turn and
# per cron fire, and a line like ``PATH=/x:${PATH}`` interpolated against an environ that already holds the
# previous reload's output grows by ``/x:`` every time until child spawns fail with E2BIG (#109902). A new
# pass resolves against the baseline instead — but only where the environ still holds exactly what we
# published, so a value the shell, config bridge, or an external secret source changed since is not frozen.
# One process-wide record (not per home/project scope): the environ is process-wide, so alternating
# callers (gateway with a project .env, cron without; home A then B) must peel each other's output too.
_DOTENV_PUBLISHED: dict[str, tuple[str | None, str, int]] = {}
_DOTENV_PASSES = itertools.count()
_DOTENV_LOCK = threading.RLock()

# Per-process credentials a parent mints and injects into the child's environment (the Desktop shell /
# a link-style launcher spawns `hermes dashboard` with a fresh HERMES_DASHBOARD_SESSION_TOKEN and keeps
# the same token for its own /api probes). They are never .env configuration, so a persisted value in
# ~/.hermes/.env must not replace an injected one — the parent would then 401 against its own child
# (#115955). A value an earlier dotenv pass published is still reloaded normally.
_SPAWN_CREDENTIAL_KEYS: frozenset[str] = frozenset({"HERMES_DASHBOARD_SESSION_TOKEN"})


def _dotenv_assignments(path: Path) -> list:
    """``(name, raw value)`` pairs of one dotenv file, parsed exactly as the loader parses it."""
    raw = path.read_bytes()
    try:
        # utf-8-sig strips a leading BOM (PowerShell 5.1 / Notepad); plain utf-8 would keep U+FEFF on the
        # first key name and silently drop it from os.environ under its canonical name.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        if raw.startswith(codecs.BOM_UTF8):  # strip the BOM by hand: utf-8-sig can't once we decode latin-1
            raw = raw[len(codecs.BOM_UTF8) :]
        text = raw.decode("latin-1")
    # Imported here, not at module level: gateway tests stub ``sys.modules["dotenv"]`` with a bare module
    # exposing only ``load_dotenv``, and ``gateway.run`` imports this module at import time.
    from dotenv.main import DotEnv

    return list(DotEnv(dotenv_path=None, stream=io.StringIO(text), interpolate=False).parse())


def _resolve_layer(assignments: list, env: dict[str, str | None], *, override: bool) -> dict[str, str]:
    """The values one dotenv layer defines, ``${VAR}`` / ``${VAR:-default}`` resolved against *env* like
    ``dotenv.main.resolve_variables`` (minus the live ``os.environ``); *override* picks which side wins a
    lookup. Callers apply the gap rule. Shared by the loader and the read-only bootstrap."""
    from dotenv.variables import parse_variables

    resolved: dict[str, str | None] = {}
    for name, value in assignments:
        if value is not None:
            lookup = {**env, **resolved} if override else {**resolved, **env}
            value = "".join(atom.resolve(lookup) for atom in parse_variables(value))
        resolved[name] = value
    return {name: value for name, value in resolved.items() if value is not None}


def _restates_published(name: str, value: str) -> bool:
    """Whether *value* is exactly what an earlier pass published for *name* and still holds: a
    gap-filling layer that states it again owns it in this pass too (so a later layer of the same
    load still sees it), without ever replacing a value someone else set. Call under ``_DOTENV_LOCK``."""
    record = _DOTENV_PUBLISHED.get(name)
    return record is not None and record[1] == value == os.environ.get(name)


def _peeled_environ(load_pass: int) -> dict[str, str | None]:
    """``os.environ`` with every value an OTHER pass published peeled back to the value it replaced,
    so ``${VAR}`` resolves against the pre-dotenv value and a reload never expands a value twice.
    Call under ``_DOTENV_LOCK``."""
    env: dict[str, str | None] = dict(os.environ)
    for name, (baseline, published, published_pass) in _DOTENV_PUBLISHED.items():
        if published_pass != load_pass and env.get(name) == published:
            if baseline is None:
                del env[name]  # absent, so ``${VAR:-default}`` takes the default again
            else:
                env[name] = baseline
    return env


def _publish_dotenv_value(name: str, value: str, load_pass: int) -> None:
    """Set one dotenv-derived value in ``os.environ`` and record it for :func:`_peeled_environ`.
    Call under ``_DOTENV_LOCK``."""
    current = os.environ.get(name)
    record = _DOTENV_PUBLISHED.get(name)
    ours = record is not None and current == record[1]
    if name in _SPAWN_CREDENTIAL_KEYS and current and not ours:
        return  # parent-minted per-process credential: .env must not split it from the parent
    # Ours and untouched since → keep the original baseline; anything else is a newer outside value.
    baseline = record[0] if ours else current
    os.environ[name] = value
    _DOTENV_PUBLISHED[name] = (baseline, value, load_pass)
