"""OAuth must not run against an unavailable or newly selected environment."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pm
import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.mark.parametrize("command", ["--check", "--check-live", "--auth-url", "--auth-code", "--revoke"])
def test_oauth_stops_at_pm_restart_boundary(command, monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_setup", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("TOKEN_PATH", "CLIENT_SECRET_PATH", "PENDING_AUTH_PATH"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"state": "pending-state", "code_verifier": "verifier"}))
        monkeypatch.setattr(module, name, path)
    before = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
    ensure = Mock(side_effect=pm.InstallError("venv", "google installed; restart Hermes to activate"))
    monkeypatch.setattr(pm, "ensure_import", ensure)
    monkeypatch.setattr("subprocess.check_call", Mock(side_effect=AssertionError("ambient install")))
    monkeypatch.setattr(sys, "argv", [str(SETUP_PATH), command] + (["code"] if command == "--auth-code" else []))

    with pytest.raises(SystemExit) as failure:
        module.main()

    assert failure.value.code == 1
    ensure.assert_called_once_with("google")
    assert "restart Hermes" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in tmp_path.glob("*.json")} == before


@pytest.mark.parametrize("command", ["--install-deps", "--auth-url"])
def test_standalone_without_hermes_reports_setup_not_ambient_installs(command, tmp_path):
    # -I -S excludes both the checkout and installed site packages, just as a
    # copied skill run with an unrelated interpreter has no Hermes PM module.
    (tmp_path / "google_client_secret.json").write_text("{}")
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(SETUP_PATH), command],
        env={**os.environ, "HERMES_HOME": str(tmp_path), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1
    assert "Hermes environment" in result.stdout
    assert "hermes setup" in result.stdout
    assert "pip" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def _load_setup_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_setup", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_absent_pm_still_runs_when_the_google_extra_is_importable(monkeypatch):
    # `sys.path[0]` is the script's own directory and the editable install does
    # not export `pm`, so a correctly installed Hermes reaches this path. The
    # Google libraries are what the command actually needs.
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "pm", None)
    monkeypatch.setattr(module, "_google_deps_importable", lambda: True)

    module._ensure_deps()


def test_absent_pm_with_missing_google_extra_still_refuses(monkeypatch, capsys):
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "pm", None)
    monkeypatch.setattr(module, "_google_deps_importable", lambda: False)

    with pytest.raises(SystemExit) as failure:
        module._ensure_deps()

    assert failure.value.code == 1
    output = capsys.readouterr().out
    assert "Hermes environment" in output
    assert "hermes setup" in output


def test_google_anchor_probe_reports_a_missing_module(monkeypatch):
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "_GOOGLE_ANCHORS", ("googleapiclient", "hermes_absent_anchor"))

    assert module._google_deps_importable() is False


def test_google_anchor_copy_matches_pm_extras(monkeypatch):
    # The copy exists because ``pm`` may be unimportable; nothing else ties the
    # two together, and a partial install must still be refused.
    module = _load_setup_module(monkeypatch)

    assert set(module._GOOGLE_ANCHORS) == set(pm.extras.ANCHORS["google"])
    assert len(module._GOOGLE_ANCHORS) == len(pm.extras.ANCHORS["google"])


@pytest.mark.parametrize("missing", range(4))
def test_google_anchor_probe_checks_every_anchor(missing, monkeypatch):
    module = _load_setup_module(monkeypatch)
    anchors = list(module._GOOGLE_ANCHORS)
    importable = set(anchors) - {anchors[missing]}

    def fake_import(name):
        if name not in importable:
            raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    assert module._google_deps_importable() is False


def test_install_deps_without_pm_succeeds_when_google_extra_is_importable(monkeypatch, capsys):
    # ``--install-deps`` is the remediation SKILL.md names; it must not refuse
    # in the same absent-``pm`` state that _ensure_deps() already accepts.
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "pm", None)
    monkeypatch.setattr(module, "_google_deps_importable", lambda: True)

    assert module.install_deps() is True
    assert "Hermes environment" not in capsys.readouterr().out


def test_install_deps_without_pm_and_without_google_extra_refuses(monkeypatch, capsys):
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr(module, "pm", None)
    monkeypatch.setattr(module, "_google_deps_importable", lambda: False)

    assert module.install_deps() is False
    output = capsys.readouterr().out
    assert "Hermes environment" in output
    assert "hermes setup" in output


class _FakeDist:
    def __init__(self, url):
        self.url = url

    def read_text(self, name):
        assert name == "direct_url.json"
        return json.dumps({"url": self.url, "dir_info": {"editable": True}})


def test_pm_is_found_through_the_recorded_checkout(monkeypatch):
    # A bundled skill is copied out of the checkout, so ``import pm`` misses it;
    # the venv's direct_url.json still names the tree it was installed from.
    module = _load_setup_module(monkeypatch)
    root = Path(pm.__file__).resolve().parents[1]
    monkeypatch.setattr("importlib.metadata.distribution",
                        lambda name: _FakeDist(root.as_uri()))
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".").resolve() != root])

    assert module._import_pm_from_checkout() is pm
    assert str(root) in sys.path


@pytest.mark.parametrize("url", ["https://example.com/hermes.tar.gz", "file:///nonexistent-hermes-root"])
def test_pm_lookup_ignores_records_without_a_checkout(url, monkeypatch):
    module = _load_setup_module(monkeypatch)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _FakeDist(url))

    assert module._import_pm_from_checkout() is None


def test_pm_lookup_without_an_installed_hermes_returns_none(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    module = _load_setup_module(monkeypatch)

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.distribution", missing)

    assert module._import_pm_from_checkout() is None
