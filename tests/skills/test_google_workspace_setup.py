"""Security-floor tests for the Google Workspace runtime installer."""

from __future__ import annotations

import importlib.util
import shlex
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)

SKILL_DOC_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/SKILL.md"
)


@pytest.fixture()
def setup_module():
    spec = importlib.util.spec_from_file_location(
        "test_google_workspace_setup_module",
        SETUP_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stale_google_transitives_are_reported_missing(setup_module, monkeypatch):
    installed = {
        "google-api-python-client": "2.194.0",
        "google-auth": "2.55.0",
        "google-auth-oauthlib": "1.3.1",
        "google-auth-httplib2": "0.3.1",
        "httplib2": "0.31.2",
        "pyasn1": "0.6.3",
    }

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(setup_module, "_distribution_version", fake_version)

    assert setup_module._missing_required_packages() == [
        "google-auth==2.55.1",
        "httplib2==0.32.0",
        "pyasn1==0.6.4",
    ]


def test_installer_repairs_stale_transitives(setup_module, monkeypatch):
    states = iter(
        [
            [
                "google-auth==2.55.1",
                "httplib2==0.32.0",
                "pyasn1==0.6.4",
            ],
            [],
        ]
    )
    monkeypatch.setattr(
        setup_module,
        "_missing_required_packages",
        lambda: next(states),
    )
    calls = []
    monkeypatch.setattr(
        setup_module.subprocess,
        "check_call",
        lambda argv, **kwargs: calls.append(argv),
    )

    assert setup_module.install_deps() is True
    assert calls == [
        [
            setup_module.sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "google-auth==2.55.1",
            "httplib2==0.32.0",
            "pyasn1==0.6.4",
        ]
    ]


def test_skill_doc_references_only_supported_setup_flags(setup_module, capsys):
    """Anti-drift guard (#35560/#74128): every $GSETUP invocation shown in
    SKILL.md must use flags setup.py actually supports.

    Before the fix, SKILL.md documented --services/--format, which setup.py
    never implemented — following the skill's auth flow failed with argparse
    'unrecognized arguments'. This test parses every documented $GSETUP line
    and fails on any flag the script doesn't define.
    """
    parser = setup_module.get_argument_parser() if hasattr(setup_module, "get_argument_parser") else None
    if parser is None:
        # Fall back to introspecting the argparse parser main() builds.
        import contextlib
        import io
        import re

        src = SETUP_PATH.read_text(encoding="utf-8")
        m = re.search(r"parser\.add_argument\(.*?--([a-z-]+)", src, re.S)
        supported = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
        assert supported, "could not extract supported flags from setup.py"
    else:
        supported = {a for a in parser._option_string_actions}

    doc_text = SKILL_DOC_PATH.read_text(encoding="utf-8")
    setup_lines = [line.strip() for line in doc_text.splitlines() if "$GSETUP" in line]
    assert setup_lines, "SKILL.md must document at least one $GSETUP invocation"

    unsupported = []
    for line in setup_lines:
        command_tail = line.split("$GSETUP", 1)[1].replace("`", "")
        args = shlex.split(command_tail)
        for arg in args:
            if arg.startswith("--") and arg not in supported:
                unsupported.append((line, arg))

    assert unsupported == [], (
        "SKILL.md references setup.py flags that don't exist: "
        f"{unsupported}. Fix the doc (or add the flag)."
    )
