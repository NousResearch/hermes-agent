"""#139085: a legacy <install>/venv/Lib/site-packages entry must not survive into sys.path."""

import sys
from pathlib import Path

from gateway import run_startup


def test_purge_drops_legacy_venv_site_packages_and_keeps_others(monkeypatch):
    install = Path(run_startup.__file__).resolve().parents[1]
    legacy = str(install / "venv" / "Lib" / "site-packages")
    keep = "/some/runtime/site-packages"
    monkeypatch.setattr(sys, "path", [legacy, legacy + "/pydantic", keep, str(install)])
    run_startup._purge_legacy_venv_sys_path()
    assert sys.path == [keep, str(install)]


def test_purge_does_not_match_sibling_prefix(monkeypatch):
    install = Path(run_startup.__file__).resolve().parents[1]
    sibling = str(install / "venv" / "Lib" / "site-packages-old")
    monkeypatch.setattr(sys, "path", [sibling])
    run_startup._purge_legacy_venv_sys_path()
    assert sys.path == [sibling]


def test_worker_start_purges_legacy_path_before_importing_tui_gateway(monkeypatch):
    import builtins

    install = Path(run_startup.__file__).resolve().parents[1]
    legacy = str(install / "venv" / "Lib" / "site-packages")
    seen = {}
    real_import = builtins.__import__

    class _Stop(Exception):
        pass

    def _fake_import(name, *a, **k):
        if name == "tui_gateway.server":
            seen["path"] = list(sys.path)
            raise _Stop
        return real_import(name, *a, **k)

    monkeypatch.setattr(sys, "path", [legacy, "/keep"])
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    try:
        run_startup.GatewayStartupMixin._start_hosted_room_worker_sync()
    except _Stop:
        pass
    assert legacy not in seen["path"]
    assert "/keep" in seen["path"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
