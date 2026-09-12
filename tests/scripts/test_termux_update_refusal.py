"""The installed-package gate distinguishes commit builds from release packages."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.update_contract import COMMIT_BUILD_UPDATE_MESSAGE


@pytest.fixture(params=["bundle", "commit-build"])
def artifact(tmp_path, request):
    root = tmp_path / "app"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({
        "source": request.param,
        "distribution": "apt-termux",
        "payload": "bundled",
        "updateMechanism": "external",
        "commit": "a" * 40,
        "tag": None if request.param == "commit-build" else "v1.2.3",
    }), encoding="utf-8")
    return root, request.param


@pytest.mark.platforms("posix")
def test_validator_accepts_real_cli_refusal_for_installed_identity(artifact, monkeypatch, capsys, tmp_path):
    from hermes_cli import main
    from scripts.termux.validate_installed import validate_update_refusal

    root, source = artifact
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    monkeypatch.setattr("hermes_cli.image_provenance.IMAGE_PROVENANCE_PATH", tmp_path / "absent")
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    def unexpected_update(*args, **kwargs):
        pytest.fail("sealed package reached the mutation path")

    monkeypatch.setattr(main, "_install_hangup_protection", unexpected_update)
    with pytest.raises(SystemExit) as stopped:
        main.cmd_update(SimpleNamespace())
    output = capsys.readouterr()
    status = stopped.value.code
    assert isinstance(status, int) and status == 2
    expected = COMMIT_BUILD_UPDATE_MESSAGE if source == "commit-build" else "pkg upgrade hermes-agent"
    assert expected in output.out + output.err
    result = subprocess.CompletedProcess(["hermes", "update"], status, output.out, output.err)
    validate_update_refusal(root, result)
    marker = "COMMIT_BUILD_UPDATE_REFUSAL_OK" if source == "commit-build" else "APT_UPDATE_REFUSAL_OK"
    assert marker in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_validator_rejects_wrong_refusal_or_exit_status(artifact):
    from scripts.termux.validate_installed import validate_update_refusal

    root, source = artifact
    expected = COMMIT_BUILD_UPDATE_MESSAGE if source == "commit-build" else "pkg upgrade hermes-agent"
    other = "pkg upgrade hermes-agent" if source == "commit-build" else COMMIT_BUILD_UPDATE_MESSAGE
    for status, message in ((0, expected), (1, expected), (2, other), (2, "unrelated startup failure")):
        result = subprocess.CompletedProcess(["hermes", "update"], status, message, "")
        with pytest.raises(RuntimeError, match="wrong updater refusal"):
            validate_update_refusal(root, result)
