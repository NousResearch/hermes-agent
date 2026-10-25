"""A per-job ``base_url`` override belongs to the provider it was pinned to (#138441).

Replacing that binding through ``update_job`` — switching providers, or releasing the pin to
follow the default route — must drop the stored endpoint too. Before the fix the Desktop model
picker saved the new provider/model while the old ``base_url`` survived the merge, so the job
stored a pair the runtime credential guard rejects only at fire time. Unrelated edits and a
rewritten-but-unchanged provider keep an intentional override; an explicit ``base_url`` in the
same update always wins.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _store(monkeypatch, tmp_path):
    from cron import jobs

    monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)
    state = {"jobs": []}
    monkeypatch.setattr(jobs, "load_jobs", lambda: list(state["jobs"]), raising=True)
    monkeypatch.setattr(jobs, "save_jobs", lambda j: state.__setitem__("jobs", list(j)), raising=True)
    monkeypatch.setattr(jobs, "resolve_job_ref", lambda ref: next(
        (j for j in state["jobs"] if j["id"] == ref), None), raising=True)
    resolver = MagicMock(return_value={"provider": "openrouter"})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)
    return jobs


def _pinned_job(jobs):
    return jobs.create_job(
        prompt="say hi",
        schedule="every 1 hour",
        model="old-model",
        provider="custom:old",
        base_url="https://old.example.invalid/v1",
    )


class TestProviderSwitchDropsOverride:
    def test_switching_provider_clears_stale_base_url(self, monkeypatch, tmp_path):
        jobs = _store(monkeypatch, tmp_path)
        job = _pinned_job(jobs)

        updated = jobs.update_job(job["id"], {"provider": "custom:new", "model": "new-model"})

        assert (updated["provider"], updated["model"]) == ("custom:new", "new-model")
        assert updated["base_url"] is None

    def test_selecting_the_default_route_clears_base_url(self, monkeypatch, tmp_path):
        jobs = _store(monkeypatch, tmp_path)
        job = _pinned_job(jobs)

        updated = jobs.update_job(job["id"], {"provider": None, "model": None})

        assert (updated["provider"], updated["model"]) == (None, None)
        assert updated["base_url"] is None

    def test_releasing_the_pin_clears_base_url(self, monkeypatch, tmp_path):
        jobs = _store(monkeypatch, tmp_path)
        job = _pinned_job(jobs)

        released = jobs.update_job(job["id"], {"pinned": False})

        assert (released["provider"], released["model"]) == (None, None)
        assert released["base_url"] is None


class TestOverrideSurvivesLegitimateKeeps:
    def test_unrelated_edit_keeps_override(self, monkeypatch, tmp_path):
        jobs = _store(monkeypatch, tmp_path)
        job = _pinned_job(jobs)

        updated = jobs.update_job(job["id"], {"name": "renamed"})

        assert updated["base_url"] == "https://old.example.invalid/v1"

    def test_same_provider_rewrite_keeps_override(self, monkeypatch, tmp_path):
        jobs = _store(monkeypatch, tmp_path)
        job = _pinned_job(jobs)

        updated = jobs.update_job(job["id"], {"provider": "custom:old", "model": "new-model"})

        assert updated["model"] == "new-model"
        assert updated["base_url"] == "https://old.example.invalid/v1"

    def test_explicit_base_url_in_same_update_wins(self, monkeypatch, tmp_path):
        jobs = _store(monkeypatch, tmp_path)
        job = _pinned_job(jobs)

        updated = jobs.update_job(job["id"], {
            "provider": "custom:new",
            "base_url": "https://new.example.invalid/v1",
        })

        assert updated["base_url"] == "https://new.example.invalid/v1"


class TestNoOverrideIsUntouched:
    def test_provider_switch_without_stored_override_adds_nothing(self, monkeypatch, tmp_path):
        jobs = _store(monkeypatch, tmp_path)
        job = jobs.create_job(prompt="say hi", schedule="every 1 hour",
                              model="old-model", provider="custom:old")

        updated = jobs.update_job(job["id"], {"provider": "custom:new", "model": "new-model"})

        assert updated["base_url"] is None
        assert "base_url" not in jobs.load_jobs()[0] or jobs.load_jobs()[0]["base_url"] is None
