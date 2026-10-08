"""Installation ownership and fail-closed scheduler state; recovery of #56787."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import update_auto_state as state


@pytest.fixture
def context(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return state.AutoUpdateContext(tmp_path / "install", home, home / "logs" / "update_receipts")


def test_unconfigured_status_is_disabled_and_read_only(context):
    before = list(context.home.rglob("*"))
    status = state.read_status(context)
    assert status["enabled"] is False
    assert status["status"] == "not_configured"
    assert list(context.home.rglob("*")) == before


def test_profiles_share_installation_identity_and_state(context, monkeypatch):
    original = context.identity
    monkeypatch.setattr(sys, "executable", "/new/python")
    monkeypatch.setattr(sys, "prefix", "/rotated/dependency/generation")
    assert context.identity == original
    other = state.AutoUpdateContext(context.install, context.home / "profiles" / "other", context.receipt_directory)
    assert other.identity == original
    assert other.status_path == context.status_path
    assert other.log_path == context.log_path
    assert other.home == context.home
    other_install = state.AutoUpdateContext(context.install / "another", context.home, context.receipt_directory)
    assert other_install.identity != original


@pytest.mark.parametrize("field,value", [("enabled", "false"), ("schema", 1),
                                       ("schedulerIdentity", "foreign"), ("dataRoot", "/other"),
                                       ("installationRoot", "/other")])
def test_untrusted_status_fails_closed(context, field, value):
    payload = {**state.default_status(context), field: value}
    context.status_path.parent.mkdir(parents=True)
    context.status_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        state.read_status(context)


@pytest.mark.parametrize("payload", ["not json", "[]", "null", "{", "\xff"])
def test_corrupt_status_is_not_silently_reset(context, payload):
    context.status_path.parent.mkdir(parents=True)
    context.status_path.write_bytes(payload.encode("latin1"))
    before = context.status_path.read_bytes()
    with pytest.raises(ValueError):
        state.read_status(context)
    assert context.status_path.read_bytes() == before


def test_state_roundtrip_and_event_log(context):
    payload = state.default_status(context)
    with state.operation_lock(context):
        state.write_status(context, payload)
        state.append_log(context, "test", detail="one\ntwo")
    assert state.read_status(context) == payload
    assert len(context.log_path.read_text().splitlines()) == 1
    assert json.loads(context.log_path.read_text())["detail"] == "one\ntwo"


@pytest.mark.platforms("posix")
def test_status_symlink_does_not_modify_target(context):
    target = context.home / "foreign"
    target.write_text("preserve")
    context.status_path.parent.mkdir(parents=True)
    context.status_path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        state.read_status(context)
    with pytest.raises(ValueError, match="symlink"):
        state.write_status(context, state.default_status(context))
    assert target.read_text() == "preserve"


def test_operation_lock_blocks_another_profile_process(context):
    code = (
        "from pathlib import Path; from hermes_cli.update_auto_state import AutoUpdateContext, operation_lock; "
        "from hermes_cli.update_lock import MarkerBusy; import sys; "
        "c=AutoUpdateContext(*map(Path,sys.argv[1:])); "
        "\ntry:\n with operation_lock(c): pass\nexcept MarkerBusy: sys.exit(7)\n"
    )
    with state.operation_lock(context):
        result = subprocess.run([sys.executable, "-c", code, str(context.install), str(context.home / "profiles" / "work"),
                                 str(context.receipt_directory)], timeout=20, capture_output=True)
    assert result.returncode == 7, result.stderr.decode()
