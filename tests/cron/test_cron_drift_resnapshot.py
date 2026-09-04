"""Drift-guard escape hatch: ``resnapshot`` on update.

The #44585 guard skips an unpinned job whose creation-time snapshot differs
from the current global config. Pins are user-owned (``hermes cron edit``),
so an agent had no way to repair the skip except delete-and-recreate, which
loses run history and metadata.

Contract:
- ``update`` with ``resnapshot=true`` adopts the CURRENT global config; the job fires.
- ``update`` without it leaves the snapshot alone; the job still skips.
- A user-set pin (programmatic ``cronjob`` caller) replaces the stale snapshot.
- A drift skip makes zero inference calls.
- The skip message names ``resnapshot=true`` and the host-side pin command.
- The agent schema still exposes no model/provider/base_url.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.cron.test_cron_provider_pin import (
    _base_job,
    _run_with_current_provider_and_model,
)

OLD_MODEL = "old-cheap-model"
NEW_MODEL = "new-premium-model"


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    # The global config the job was created under.
    (hermes_home / "config.yaml").write_text(f"model:\n  default: {OLD_MODEL}\n")
    return hermes_home


def _create_drifted_job(cron_env):
    """Create an unpinned job under OLD_MODEL, then move global config to NEW_MODEL."""
    from cron.jobs import create_job

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value={"provider": "openrouter"},
    ):
        job = create_job(prompt="hello", schedule="every 5m", deliver="local")
    assert job["model_snapshot"] == OLD_MODEL
    (cron_env / "config.yaml").write_text(f"model:\n  default: {NEW_MODEL}\n")
    return job


def _fire(job_id, tmp_path):
    from cron.jobs import get_job

    job = _base_job(**get_job(job_id))
    return _run_with_current_provider_and_model(job, "openrouter", NEW_MODEL, tmp_path)


def _update(**kwargs):
    from tools.cronjob_tools import cronjob

    return json.loads(cronjob(action="update", **kwargs))


def test_programmatic_pin_clears_stale_snapshot_and_fires(cron_env, tmp_path):
    """Pins stay user-owned; a direct cronjob() caller (CLI/dashboard) pins."""
    from cron.jobs import get_job

    job = _create_drifted_job(cron_env)
    result = _update(job_id=job["id"], provider="openrouter", model=NEW_MODEL)
    assert result["success"], result
    stored = get_job(job["id"])
    assert stored["model"] == NEW_MODEL
    assert stored["model_snapshot"] is None
    assert stored["provider_snapshot"] is None

    success, _out, _final, error, agent_constructed = _fire(job["id"], tmp_path)
    assert agent_constructed is True
    assert success is True, error


def test_update_with_resnapshot_adopts_current_config(cron_env, tmp_path):
    from cron.jobs import get_job

    job = _create_drifted_job(cron_env)
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value={"provider": "openrouter"},
    ):
        result = _update(job_id=job["id"], resnapshot=True)
    assert result["success"], result
    stored = get_job(job["id"])
    assert stored["model"] is None, "resnapshot must not pin"
    assert stored["model_snapshot"] == NEW_MODEL
    assert "resnapshot" not in stored

    success, _out, _final, error, agent_constructed = _fire(job["id"], tmp_path)
    assert agent_constructed is True
    assert success is True, error


def test_resnapshot_and_pin_are_mutually_exclusive(cron_env):
    job = _create_drifted_job(cron_env)
    result = _update(job_id=job["id"], resnapshot=True, model=NEW_MODEL)
    assert result["success"] is False
    assert "mutually exclusive" in result["error"]


def test_resnapshot_rejected_on_pinned_job(cron_env):
    from cron.jobs import create_job

    job = create_job(prompt="hello", schedule="every 5m", deliver="local", model="pinned")
    result = _update(job_id=job["id"], resnapshot=True)
    assert result["success"] is False
    assert "pinned" in result["error"]


def test_update_without_pin_or_resnapshot_still_skips(cron_env, tmp_path):
    from cron.jobs import get_job

    job = _create_drifted_job(cron_env)
    result = _update(job_id=job["id"], name="renamed")
    assert result["success"], result
    assert get_job(job["id"])["model_snapshot"] == OLD_MODEL

    success, output, _final, error, agent_constructed = _fire(job["id"], tmp_path)
    assert agent_constructed is False, "drift skip must make zero inference calls"
    assert success is False
    blob = f"{error}\n{output}"
    assert "drifted" in blob
    assert f"cronjob_manage action=update job_id={job['id']} resnapshot=true" in blob
    assert f"hermes cron edit {job['id']} --provider <provider> --model <model>" in blob
    assert "cronjob action=update" not in blob


def test_schema_exposes_resnapshot_only():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    props = CRONJOB_SCHEMA["parameters"]["properties"]
    assert props["resnapshot"]["type"] == "boolean"
    for user_owned in ("model", "provider", "base_url"):
        assert user_owned not in props


def test_registry_handler_forwards_resnapshot_only():
    import tools.cronjob_tools  # noqa: F401  (registers the tool)
    from tools.registry import registry

    handler = registry.get_entry("cronjob_manage").handler
    with patch("tools.cronjob_tools.cronjob", return_value="{}") as fn:
        handler({"action": "update", "job_id": "j", "model": "m", "provider": "p", "resnapshot": True})
    kwargs = fn.call_args.kwargs
    assert kwargs["resnapshot"] is True
    assert "model" not in kwargs and "provider" not in kwargs
