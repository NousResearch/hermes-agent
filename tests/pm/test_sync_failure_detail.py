"""A refused sync retains an actionable reason in its persisted receipt."""
import importlib

import pytest


def test_lazy_refusal_records_reason_without_installing(tmp_path, monkeypatch):
    from pm import receipt
    from pm.package import InstallError

    ensure = importlib.import_module("pm.ensure")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: False)
    monkeypatch.setattr("pm.features.read_features", lambda: ["base"])

    with pytest.raises(InstallError):
        ensure.sync_venv(["new-extra"])

    saved = receipt.latest()
    assert saved["refusal"]["code"] == "lazy-install"
    assert saved["outcome"] == "failed"
    assert saved["exit_code"] != 0
    assert saved["steps"][-1]["ok"] is False
    assert "new-extra" in saved["steps"][-1]["detail"]
    assert "hermes pm install" in saved["steps"][-1]["detail"]
