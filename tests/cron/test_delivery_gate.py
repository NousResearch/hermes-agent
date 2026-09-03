"""Tests for the cron paging allowlist (cron/delivery_gate.py).

Behavior contracts: the gate must never suppress non-chat targets, must fail
closed when the allowlist file is missing, and must reload the allowlist on
every check so out-of-band edits take effect without a restart.
"""

import pytest

from cron.delivery_gate import (
    _default_settings,
    delivery_gate_check,
    suppress_and_audit,
)


def _settings(tmp_path, enabled=True, **overrides):
    base = {
        "enabled": enabled,
        "allowlist_path": str(tmp_path / "page-allowlist.txt"),
        "audit_log_path": str(tmp_path / "alerts" / "delivery-gate.log"),
        "output_root": str(tmp_path / "output"),
    }
    base.update(overrides)
    return base


def _job(name="nightly-summary", job_id="job-1"):
    return {"name": name, "id": job_id, "deliver": "origin"}


def _chat_targets():
    return [{"platform": "telegram", "chat_id": "123"}]


def test_disabled_gate_never_suppresses(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    allowed, reason = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is True
    assert reason == "gate disabled"


def test_allowlisted_name_passes(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "page-allowlist.txt").write_text("nightly-summary\n", encoding="utf-8")
    allowed, _ = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is True


def test_allowlisted_id_passes(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "page-allowlist.txt").write_text("job-1\n", encoding="utf-8")
    allowed, _ = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is True


def test_unlisted_chat_job_is_suppressed(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "page-allowlist.txt").write_text("other-job\n", encoding="utf-8")
    allowed, reason = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is False
    assert reason == "not in page-allowlist"


def test_non_chat_targets_are_never_suppressed(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "page-allowlist.txt").write_text("other-job\n", encoding="utf-8")
    targets = [{"platform": "local", "chat_id": ""}, {"platform": "", "chat_id": ""}]
    allowed, reason = delivery_gate_check(_job(), targets, settings)
    assert allowed is True
    assert reason == "no chat target"


def test_missing_allowlist_fails_closed(tmp_path):
    settings = _settings(tmp_path)  # allowlist file was never created
    allowed, reason = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is False
    assert reason == "not in page-allowlist"


def test_allowlist_ignores_comments_and_blanks(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "page-allowlist.txt").write_text(
        "# paging allowlist\n\nnightly-summary\n", encoding="utf-8"
    )
    allowed, _ = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is True


def test_allowlist_reloaded_on_every_check(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "page-allowlist.txt").write_text("", encoding="utf-8")
    allowed, _ = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is False
    # Out-of-band edit takes effect without a restart.
    (tmp_path / "page-allowlist.txt").write_text("nightly-summary\n", encoding="utf-8")
    allowed, _ = delivery_gate_check(_job(), _chat_targets(), settings)
    assert allowed is True


def test_suppress_and_audit_writes_log_and_payload(tmp_path):
    settings = _settings(tmp_path)
    suppress_and_audit(_job(), _chat_targets(), "payload body", settings)
    audit = (tmp_path / "alerts" / "delivery-gate.log").read_text(encoding="utf-8")
    assert audit.startswith("SUPPRESS ")
    assert "job=job-1" in audit
    assert "name=nightly-summary" in audit
    assert "target=telegram:123" in audit
    saved = list((tmp_path / "output" / "job-1").glob("*.md"))
    assert len(saved) == 1
    assert "payload body" in saved[0].read_text(encoding="utf-8")


def test_suppress_audit_sanitizes_job_id_for_path(tmp_path):
    settings = _settings(tmp_path)
    suppress_and_audit(_job(job_id="../evil"), _chat_targets(), "x", settings)
    # Slashes become underscores; the result stays inside output_root.
    assert (tmp_path / "output" / ".._evil").is_dir()
    assert not (tmp_path / "evil").exists()


def test_suppress_audit_dot_only_ids_stay_inside_root(tmp_path):
    settings = _settings(tmp_path)
    suppress_and_audit(_job(job_id=".."), _chat_targets(), "x", settings)
    suppress_and_audit(_job(job_id="."), _chat_targets(), "x", settings)
    assert (tmp_path / "output" / "_..").is_dir()
    assert (tmp_path / "output" / "_.").is_dir()
    # The payload never lands outside the output root.
    assert not (tmp_path / ".." / "job-1").exists()


def test_config_propagation_via_load_config(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    (tmp_path / "page-allowlist.txt").write_text("nightly-summary\n", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"cron": {"paging": settings}},
    )
    allowed, _ = delivery_gate_check(_job(), _chat_targets())
    assert allowed is True


def test_default_settings_are_disabled():
    defaults = _default_settings()
    assert defaults["enabled"] is False
    assert "page-allowlist.txt" in defaults["allowlist_path"]
