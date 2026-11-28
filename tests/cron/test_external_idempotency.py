"""Store-local creation retries, including independent Python processes."""
import importlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from cron import jobs


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path):
        yield tmp_path


def plan(**changes) -> dict[str, Any]:
    return dict(prompt="synthetic payload", schedule="every 2h",
                external_idempotency_key="automation:opaque-1", **changes)


def test_replay_edit_pause_resume_and_legacy(store):
    args = plan()
    original = jobs.create_job(**args)
    assert jobs.create_job(**args)["id"] == original["id"]
    jobs.pause_job(original["id"])
    assert jobs.create_job(**args)["state"] == "paused"
    jobs.resume_job(original["id"])
    jobs.update_job(original["id"], {"prompt": "edited", "schedule": "every 3h"})
    replay = jobs.create_job(**args)
    assert replay["prompt"] == "edited"
    assert replay["external_idempotency_fingerprint"] == original["external_idempotency_fingerprint"]
    assert len(jobs.load_jobs()) == 1
    for changes in ({"prompt": "different"}, {"schedule": "every 3h"}, {"paused": True},
                    {"origin": {"chat_id": "synthetic"}}, {"enabled_toolsets": []}):
        with pytest.raises(ValueError, match="^cron_idempotency_conflict$"):
            jobs.create_job(**(args | changes))
    legacy = jobs.create_job("legacy", "every 2h")
    assert "external_idempotency_key" not in legacy
    assert jobs.create_job("legacy", "every 2h")["id"] != legacy["id"]
    assert jobs.remove_job(original["id"])
    assert jobs.create_job(**args)["id"] != original["id"]


def test_reload_and_elapsed_oneshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    importlib.reload(jobs)
    instant = jobs._hermes_now()
    args = plan() | {"schedule": (instant + timedelta(hours=1)).isoformat()}
    original = jobs.create_job(**args)
    importlib.reload(jobs)
    with jobs.use_cron_store(tmp_path):
        monkeypatch.setattr(jobs, "_hermes_now", lambda: instant + timedelta(days=2))
        assert jobs.create_job(**args)["id"] == original["id"]
        with pytest.raises(ValueError):
            jobs.create_job(**(args | {"external_idempotency_key": None}))
        assert len(jobs.load_jobs()) == 1


@pytest.mark.parametrize("key", ["", " space", "space ", "line\n", "a/b", "é", "a" * 129, 7])
def test_invalid_key_is_not_normalized(store, key):
    with pytest.raises(ValueError, match="^cron_idempotency_invalid_key$"):
        jobs.create_job(**(plan() | {"external_idempotency_key": key}))
    assert jobs.load_jobs() == []


def test_maximum_ascii_key_and_wrapper(store):
    from cron.scheduler import create_job_with_scheduler_registration

    args = plan() | {"external_idempotency_key": "Aa09._:-" * 16, "paused": True}
    first = create_job_with_scheduler_registration(**args)
    assert create_job_with_scheduler_registration(**args)["id"] == first["id"]
    assert len(jobs.load_jobs()) == 1


@pytest.mark.parametrize("outcome", [None, False, OSError("synthetic lock failure")])
def test_degraded_lock_refuses_keyed_creation(store, monkeypatch, outcome):
    def acquire(*args):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(jobs, "_acquire_flock", acquire)
    with pytest.raises(RuntimeError, match="^cron_idempotency_lock_unavailable$"):
        jobs.create_job(**plan())
    assert jobs.load_jobs() == []


def test_parallel_processes_and_fresh_process_readback(store):
    source = (
        "import json, sys; from cron.jobs import create_job; "
        "sys.stdin.readline(); print(json.dumps(create_job(**json.loads(sys.argv[1]))))"
    )
    env = dict(os.environ, HERMES_HOME=str(store),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    processes = [subprocess.Popen([sys.executable, "-c", source, json.dumps(plan())],
                                 env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True) for _ in range(6)]
    try:
        for child in processes:
            child.stdin.write("go\n")
            child.stdin.flush()
        records = []
        for child in processes:
            stdout, stderr = child.communicate(timeout=60)
            assert child.returncode == 0, stderr
            records.append(json.loads(stdout))
        assert len({record["id"] for record in records}) == 1
        assert len(jobs.load_jobs()) == 1
        fresh = subprocess.run([sys.executable, "-c", source, json.dumps(plan())],
                               input="go\n", env=env, capture_output=True, text=True,
                               timeout=60, check=True)
        assert json.loads(fresh.stdout)["id"] == records[0]["id"]
    finally:
        for child in processes:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


@pytest.mark.parametrize("field", ["external_idempotency_key", "external_idempotency_fingerprint"])
def test_creation_identity_metadata_is_immutable(store, field):
    original = jobs.create_job(**plan())
    with pytest.raises(ValueError, match="cannot be updated"):
        jobs.update_job(original["id"], {field: None})
    assert jobs.create_job(**plan())["id"] == original["id"]
    assert len(jobs.load_jobs()) == 1


def test_keyed_creation_rejects_lock_for_another_store(tmp_path, monkeypatch):
    first = tmp_path / "profile-a"
    second = tmp_path / "profile-b"
    monkeypatch.setenv("HERMES_HOME", str(first))
    with jobs.use_cron_store(first), jobs._jobs_lock():
        with jobs.use_cron_store(second):
            with pytest.raises(RuntimeError, match="^cron_idempotency_lock_unavailable$"):
                jobs.create_job(**plan())
            assert jobs.load_jobs() == []


def test_transport_origin_facts_do_not_split_a_retry(store):
    first = dict(platform="api_server", chat_id="api", source_ip="198.51.100.7", user_agent="client/1")
    retry = dict(platform="api_server", chat_id="api", source_ip="203.0.113.9", forwarded_for="203.0.113.9",
                 user_agent="client/2")
    original = jobs.create_job(**plan(origin=first))
    assert jobs.create_job(**plan(origin=retry))["id"] == original["id"]
    with pytest.raises(ValueError, match="^cron_idempotency_conflict$"):
        jobs.create_job(**plan(origin=dict(retry, chat_id="another-chat")))
    assert len(jobs.load_jobs()) == 1
