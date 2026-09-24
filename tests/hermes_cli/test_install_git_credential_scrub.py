"""#101351 — the installers' git clone failures must not echo the source URL.

``mcp_catalog._do_git_install`` printed git's stderr raw and raised
``CatalogError(f"git clone failed for {install.url}")``;
``profile_distribution._git_clone`` raised ``DistributionError`` with the raw
stderr. A credentialed manifest/profile source therefore put the PAT on the
console, in a log, and in whatever captured the traceback.
"""
from __future__ import annotations

import pytest

TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0"
CREDENTIALED = f"https://x-access-token:{TOKEN}@127.0.0.1:9/acme/private.git"
STDERR = (
    "Cloning into 'dest'...\n"
    f"fatal: unable to access '{CREDENTIALED}/': Failed to connect to 127.0.0.1 port 9\n"
)

# Both call sites late-import this name from hermes_cli.git_credentials, so the
# patch has to land on the source module for the lazy import to see it.
_TARGET = "hermes_cli.git_credentials.run_git_with_credential_fallback"


def _fake_result(*args, **kwargs):
    return type("R", (), {"returncode": 128, "stderr": STDERR})()


def test_profile_distribution_clone_failure_scrubs_the_url(tmp_path, monkeypatch) -> None:
    from hermes_cli import profile_distribution as pd

    monkeypatch.setattr(_TARGET, _fake_result)
    with pytest.raises(pd.DistributionError) as exc:
        pd._git_clone(CREDENTIALED, tmp_path / "dest")
    assert TOKEN not in str(exc.value)
    assert "x-access-token" not in str(exc.value)
    assert "Failed to connect" in str(exc.value)  # the diagnosis survives


def test_mcp_catalog_clone_failure_scrubs_the_url(tmp_path, monkeypatch, capsys) -> None:
    from hermes_cli import mcp_catalog as mc

    monkeypatch.setattr(_TARGET, _fake_result)
    monkeypatch.setattr(mc, "_install_root", lambda: tmp_path)
    install = type("Inst", (), {"type": "git", "url": CREDENTIALED, "ref": "a" * 40, "bootstrap": None})()
    entry = type("Entry", (), {"name": "probe", "install": install})()
    with pytest.raises(mc.CatalogError) as exc:
        mc._do_git_install(entry)
    assert TOKEN not in str(exc.value)
    assert "x-access-token" not in str(exc.value)
    # The progress line and the printed stderr must not carry it either.
    printed = capsys.readouterr().out
    assert TOKEN not in printed
    assert "x-access-token" not in printed
