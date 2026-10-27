"""Stage publishing into a watching desktop-update UI (hermes_cli.update_stage).

The desktop hand-off shim renders progress from a status JSON file. The
regression: during an old→new checkout transition the OLD shim never exports
the status path, so the takeover children (PM sync, builds) ran for minutes
while the shim's UI sat frozen on its last stage — the user's only signal
that anything was happening. update_stage must discover the file through
both the exported env var (new shim) and the marker-pid fallback (old shim),
and must be inert when neither exists (plain CLI update, no UI at all).
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from hermes_cli import update_stage


@pytest.fixture
def status_file(tmp_path):
    return tmp_path / "hermes-update-status.80335"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(update_stage.STATUS_FILE_ENV, raising=False)
    monkeypatch.delenv(update_stage.UI_SPAWNED_ENV, raising=False)
    # gettempdir() caches its first answer per process; the shim writes beside ${TMPDIR}
    # so tests that redirect TMPDIR need the cache cleared to be observed.
    monkeypatch.setattr(tempfile, "tempdir", None)


def _assert_running(payload: str, message: str) -> None:
    data = json.loads(payload)
    assert data["status"] == "running", "only the shim publishes terminal states"
    assert data["message"] == message


def test_publishes_through_exported_env_var(status_file, monkeypatch):
    monkeypatch.setenv(update_stage.STATUS_FILE_ENV, str(status_file))

    update_stage.publish_stage("Building the web UI")

    _assert_running(status_file.read_text(encoding="utf-8"), "Building the web UI")


def test_marker_fallback_covers_the_old_shim(status_file, tmp_path, monkeypatch):
    """The old→new transition: no env var, but the marker names the shim pid.

    The shim's status file is ${TMPDIR}/hermes-update-status.<shim pid> and
    the marker's first line IS that pid (update_lock adopts, never rewrites).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / update_stage.MARKER_NAME).write_text(
        f"80335\n{int(time.time())}\n", encoding="utf-8")
    status_file.write_text('{"status":"running","message":"old"}', encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    update_stage.publish_stage("Updating Python dependencies (PM)")

    _assert_running(status_file.read_text(encoding="utf-8"),
                    "Updating Python dependencies (PM)")


def test_marker_fallback_uses_the_platform_home_without_env_var(status_file, tmp_path, monkeypatch):
    """The shim without HERMES_HOME resolved the platform default, which is not ~/.hermes
    on every host (sudo invoker, data-dir suffix); the marker must be looked up there."""
    import hermes_constants

    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: tmp_path / "platform")
    (tmp_path / "platform").mkdir()
    (tmp_path / "platform" / update_stage.MARKER_NAME).write_text(
        f"80335\n{int(time.time())}\n", encoding="utf-8")
    status_file.write_text('{"status":"running","message":"old"}', encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    update_stage.publish_stage("Building products")

    _assert_running(status_file.read_text(encoding="utf-8"), "Building products")


def test_no_ui_sources_is_inert(status_file, tmp_path, monkeypatch):
    """A plain CLI update has no env var and no marker: publish must no-op."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no marker inside
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    update_stage.publish_stage("anything")

    assert not status_file.exists()


def test_publish_never_raises_on_unwritable_target(monkeypatch, tmp_path):
    monkeypatch.setenv(update_stage.STATUS_FILE_ENV, str(tmp_path / "no" / "such" / "dir" / "f"))

    update_stage.publish_stage("boom")  # must not raise


def test_ensure_panel_skipped_off_macos(status_file, tmp_path, monkeypatch):
    """ensure_panel is a macOS-only fallback; elsewhere it must not spawn."""
    monkeypatch.setenv(update_stage.STATUS_FILE_ENV, str(status_file))
    called = []
    monkeypatch.setattr(update_stage, "_ui_present_in_log", lambda text: False)

    # os.uname() on Linux reports Linux; the guard must reject before spawn.
    update_stage.ensure_panel(tmp_path)

    assert called == []
    assert update_stage.UI_SPAWNED_ENV not in os.environ
