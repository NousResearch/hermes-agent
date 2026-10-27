import json

import pytest

from hermes_cli.backup import restore_cron_jobs_if_emptied
from hermes_cli.backup_cron_prompts import restore_cron_prompt_fields_if_degraded


@pytest.mark.parametrize("restore", [restore_cron_jobs_if_emptied, restore_cron_prompt_fields_if_degraded])
@pytest.mark.parametrize("damaged_file", ["live", "snapshot"])
def test_cron_recovery_preserves_undecodable_files(tmp_path, restore, damaged_file):
    live_path = tmp_path / "cron" / "jobs.json"
    snapshot_path = tmp_path / "state-snapshots" / "before-update" / "cron" / "jobs.json"
    live_path.parent.mkdir(parents=True)
    snapshot_path.parent.mkdir(parents=True)
    live_path.write_text('{"jobs": []}', encoding="utf-8")
    snapshot_path.write_text(
        '{"jobs": [{"id": "morning", "name": "Morning", "prompt": "Read the news."}]}',
        encoding="utf-8",
    )
    if damaged_file == "live":
        live_path.write_bytes(b'{"jobs": [\xff]}')
    else:
        snapshot_path.write_bytes(b'{"jobs": [\xff]}')
    live_before = live_path.read_bytes()
    snapshot_before = snapshot_path.read_bytes()

    assert restore("before-update", hermes_home=tmp_path) is None

    assert live_path.read_bytes() == live_before
    assert snapshot_path.read_bytes() == snapshot_before


@pytest.mark.parametrize("restore", [restore_cron_jobs_if_emptied, restore_cron_prompt_fields_if_degraded])
@pytest.mark.parametrize("legacy_snapshot", [False, True])
def test_cron_recovery_restores_bom_unicode_snapshot(tmp_path, restore, legacy_snapshot):
    job = {
        "id": "morning",
        "name": "Café matinal",
        "prompt": "Résume les nouvelles du matin.",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "enabled": True,
    }
    live_path = tmp_path / "cron" / "jobs.json"
    snapshot_path = tmp_path / "state-snapshots" / "before-update" / "cron" / "jobs.json"
    live_path.parent.mkdir(parents=True)
    snapshot_path.parent.mkdir(parents=True)
    snapshot = [job] if legacy_snapshot else {"jobs": [job]}
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8-sig")
    live_jobs = []
    if restore is restore_cron_prompt_fields_if_degraded:
        live_jobs = [{**job, "prompt": job["name"]}]
    live_path.write_text(json.dumps({"jobs": live_jobs}), encoding="utf-8")

    result = restore("before-update", hermes_home=tmp_path)

    assert result is not None and result["restored"]
    restored = json.loads(live_path.read_text(encoding="utf-8-sig"))
    restored_jobs = restored if isinstance(restored, list) else restored["jobs"]
    assert len(restored_jobs) == 1
    assert restored_jobs[0]["prompt"] == "Résume les nouvelles du matin."
    assert restored_jobs[0]["name"] == "Café matinal"
