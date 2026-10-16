"""Webapp host shells start with a locale the host's C library accepts.

glibc rejects the bare charset name ``UTF-8`` that macOS accepts as a locale, and
LC_CTYPE outranks LANG, so forcing ``LC_CTYPE=UTF-8`` on Linux made every host
shell warn and fall back to ASCII even when the user's LANG was valid.
"""
from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.parametrize(
    ("env", "platform", "expected"),
    [
        # Linux: a set LANG already governs LC_CTYPE; only a missing one needs C.UTF-8.
        ({"LANG": "en_US.UTF-8"}, "linux", None),
        ({}, "linux", "C.UTF-8"),
        ({"LANG": ""}, "linux", "C.UTF-8"),
        # An explicit LC_CTYPE or LC_ALL is the user's choice on every platform.
        ({"LANG": "en_US.UTF-8", "LC_CTYPE": "de_DE.UTF-8"}, "linux", None),
        ({"LC_ALL": "de_DE.UTF-8"}, "linux", None),
        ({"LC_CTYPE": "de_DE.UTF-8"}, "darwin", None),
        # macOS accepts the bare charset, and a launchd-started backend often has no LANG.
        ({}, "darwin", "UTF-8"),
        ({"LANG": "en_US.UTF-8"}, "darwin", "UTF-8"),
        # Windows shells take their code page from the console, not LC_CTYPE.
        ({}, "win32", None),
    ],
)
def test_lc_ctype_rule(env, platform, expected):
    from hermes_cli.web_host_terminal import terminal_lc_ctype

    assert terminal_lc_ctype(env, platform) == expected


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("lang", ["C.UTF-8", None], ids=["lang-set", "lang-unset"])
def test_linux_host_shell_env_selects_a_utf8_locale(monkeypatch, lang):
    """The env resolve_argv hands the PTY must pass glibc's setlocale with a UTF-8 codeset."""
    from hermes_cli.web_host_terminal import resolve_argv
    from hermes_constants import get_hermes_home

    for key in [k for k in os.environ if k.startswith("LC_") or k in ("LANG", "LANGUAGE")]:
        monkeypatch.delenv(key)
    if lang is not None:
        monkeypatch.setenv("LANG", lang)

    _argv, _cwd, env, _shell = resolve_argv(home=get_hermes_home())
    probe = subprocess.run(
        ["locale", "charmap"], env=env, capture_output=True, text=True, timeout=30, check=False,
    )

    assert "Cannot set" not in probe.stderr, probe.stderr
    assert probe.stdout.strip() == "UTF-8"
    if lang is not None:
        assert env["LANG"] == lang
