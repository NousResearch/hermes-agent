"""External cron workers identify themselves in cron failure notices.

The failure-notice emitter tag reads the process's boot fingerprint
(``gateway.code_skew.boot_code_sha``). Gateway processes snapshot it at startup, but an
external worker (``python -m cron.scheduler --external-worker-file``) never runs gateway
startup, so every notice emitted from the worker used to say ``loaded_revision=unknown``
even when its checkout revision was readable — leaving the operator unable to tell which
code the worker loaded. The worker records its boot fingerprint before running the job.
"""
import json

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A launch home and an owning profile home, neither with a hooks config."""
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    profile = tmp_path / "profile"
    profile.mkdir()
    yield launch, profile


def test_external_worker_records_its_boot_revision(homes, tmp_path, monkeypatch):
    import cron.scheduler as scheduler
    import gateway.code_skew as code_skew

    launch, profile = homes
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"job": {"id": "job-1", "execution_id": "exec-1"},
                    "profile_home": str(profile), "multiplex_active": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    monkeypatch.setattr(scheduler, "run_one_job", lambda *a, **k: True)

    assert scheduler._run_external_worker_payload(payload, tmp_path / "exec-1.ready") is True
    # The worker snapshots its checkout revision: the emitter tag names the loaded code,
    # falling back to ``unknown`` only when the revision is truly unreadable.
    assert code_skew.boot_code_sha() is not None
