"""The SQLite runtime repair outcome must reach the persisted update receipt (#111497).

``record_step``/``record_skip`` are the only way into a receipt; before this a failed repair left
``outcome: partial`` with no step naming the reason or the SQLite versions involved.
"""

import json

import pytest

from hermes_cli import managed_uv as uv
from hermes_cli import update_receipt as receipts


@pytest.mark.parametrize("status, ok, bucket", [
    ("failed", False, "steps"),
    ("repaired", True, "steps"),
    ("skipped", None, "skips"),
])
def test_runtime_repair_outcome_reaches_persisted_receipt(tmp_path, monkeypatch, status, ok, bucket):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(receipts, "_current", None)
    result = uv.RuntimeRepairResult(
        status, "candidate dependency sync failed (rc=1): error: lockfile stale", "3.50.4", "3.53.1")
    monkeypatch.setattr(uv, "repair_vulnerable_runtime", lambda _: result)
    monkeypatch.setattr(uv, "resolve_uv", lambda: "uv")
    monkeypatch.setattr(uv, "_uv_self_update_is_fresh", lambda: True)
    observed = []

    receipts.begin_update_receipt()
    uv.update_managed_uv(repair_observer=observed.append)
    data = json.loads(receipts.finalize_update_receipt("partial").read_text(encoding="utf-8"))

    assert observed == [result]
    entry, = data[bucket]
    other = "skips" if bucket == "steps" else "steps"
    assert data[other] == []
    assert entry["name"] == "sqlite_runtime_repair"
    text = entry["detail"] if bucket == "steps" else entry["reason"]
    if bucket == "steps":
        assert entry["ok"] is ok
    assert text.startswith(f"{status}: ")
    assert result.detail in text
    assert "sqlite 3.50.4 → 3.53.1" in text
    # Outside `hermes update` (setup/bootstrap) there is no receipt; the hook must stay a no-op.
    uv._run_runtime_repair("uv", observed.append)
    assert receipts._current is None
    assert observed == [result, result]
