"""Children the suite spawns keep its no-fetch rule even when their environment is rebuilt.

Many tests launch a real ``gateway.run`` or ``hermes_cli.main`` with an env built from scratch
(``{k: os.environ[k] for k in ("PATH", "LANG", "TZ")}``) or with every ``HERMES_*`` variable
scrubbed, so the child looks like a real install. That also drops ``HERMES_DISABLE_LAZY_INSTALLS``,
and a fresh home then boots like a user's: the boot step spawns the detached default-plugins
installer, which clones every ``default: true`` catalog plugin from GitHub, installs its Python
dependencies and provisions a toolchain under ``HERMES_HOME/tools``. That detached child outlives
the test and writes into the temp home while it is removed (``[Errno 39] Directory not empty``).

Every ``subprocess.Popen`` (``run``, ``check_output`` and asyncio's subprocess transport go through
it) given an explicit ``env`` without the kill-switch gets the parent's value. A test that has
lifted the switch in its own process (``monkeypatch.delenv``) is left alone, and a child that must
fetch for real says so with an explicit value (``HERMES_DISABLE_LAZY_INSTALLS="0"``).
"""

from __future__ import annotations

import functools
import os
import subprocess

KILL_SWITCH = "HERMES_DISABLE_LAZY_INSTALLS"


def with_parent_kill_switch(env):
    """*env* plus the parent's kill-switch when *env* does not set the switch itself."""
    value = os.environ.get(KILL_SWITCH)
    if env is None or not value:
        return env
    as_bytes = any(isinstance(key, bytes) for key in env)
    key = os.fsencode(KILL_SWITCH) if as_bytes else KILL_SWITCH
    if key in env:
        return env
    return {**env, key: os.fsencode(value) if as_bytes else value}


def install() -> None:
    original = subprocess.Popen.__init__

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if "env" in kwargs:
            kwargs["env"] = with_parent_kill_switch(kwargs["env"])
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__
