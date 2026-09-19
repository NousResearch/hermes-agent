"""GitHub auth probes must select a usable active account across gh versions."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli.doctor_state import _gh_authenticated


JSON_PROBE = ["gh", "auth", "status", "--json", "hosts", "--hostname", "github.com"]
TEXT_PROBE = ["gh", "auth", "status", "--hostname", "github.com"]


@pytest.mark.parametrize(
    "accounts, expected",
    [
        ([{"active": True, "state": "success"}], True),
        ([{"active": False, "state": "success"}], False),
        ([{"active": True, "state": "failed"}], False),
        ([{"state": "success"}], False),
        ([{"active": "false", "state": "success"}], False),
        ([{"active": True, "state": "failed"},
          {"active": False, "state": "success"}], False),
        ([{"active": False, "state": "failed"},
          {"active": True, "state": "success"}], True),
        ([None, 42, "account", [], {"active": True, "state": "success"}], True),
        ([None, 42, "account", []], False),
        ([], False),
    ],
)
def test_json_requires_usable_active_account(monkeypatch, accounts, expected):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["timeout"] == 10
        assert kwargs["capture_output"] is True
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"hosts": {"github.com": accounts}}),
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert _gh_authenticated() is expected
    # A valid JSON verdict must not be overridden by text-mode exit status.
    assert calls == [JSON_PROBE]


@pytest.mark.parametrize(
    "first",
    [
        (1, "unknown flag: --json"),
        (0, "not json"),
        (0, ""),
        (0, "null"),
        (0, "42"),
        (0, '"account"'),
        (0, "[]"),
        (0, '{"hosts": null}'),
        (0, '{"hosts": []}'),
        (0, '{"hosts": {"github.com": null}}'),
        (0, '{"hosts": {"github.com": 42}}'),
        (0, '{"hosts": {"github.com": "account"}}'),
        (0, '{"hosts": {"github.com": {"active": true, "state": "success"}}}'),
        FileNotFoundError(),
        PermissionError(),
        subprocess.TimeoutExpired(JSON_PROBE, 10),
    ],
)
@pytest.mark.parametrize(
    "fallback",
    [0, 1, FileNotFoundError(), PermissionError(), subprocess.TimeoutExpired(TEXT_PROBE, 10)],
)
def test_unavailable_json_uses_bounded_nonfatal_legacy_probe(monkeypatch, first, fallback):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["timeout"] == 10
        assert kwargs["capture_output"] is True
        if "--json" in cmd:
            if isinstance(first, Exception):
                raise first
            return SimpleNamespace(returncode=first[0], stdout=first[1])
        assert cmd == TEXT_PROBE
        if isinstance(fallback, Exception):
            raise fallback
        return SimpleNamespace(returncode=fallback)

    monkeypatch.setattr(subprocess, "run", run)
    expected = not isinstance(first, Exception) and fallback == 0
    assert _gh_authenticated() is expected
    assert calls == ([JSON_PROBE] if isinstance(first, Exception) else [JSON_PROBE, TEXT_PROBE])
