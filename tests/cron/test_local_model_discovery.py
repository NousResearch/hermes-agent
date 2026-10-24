"""Cron local-model discovery (issue #20125): unpinned jobs on a loopback
inference server run on what is actually loaded, instead of failing fast.

Contract, mirroring the interactive side's existing behavior:
- discovery fires ONLY when the model chain (job pin > cron.model > main
  agent model) resolved to empty;
- the server is asked ONLY on loopback base_urls;
- exactly-one-model answers are accepted; ambiguous answers are not;
- any probe failure is absorbed (job then fails fast as before);
- a resolved model is used, logged once, and the existing fail-fast
  raise still fires when discovery yields nothing.
"""

from types import SimpleNamespace
from unittest import mock

import pytest


# --------------------------------------------------------------------------- helpers


def _discovery():
    from cron import scheduler_local_discovery as sld

    return sld


# --------------------------------------------------------------------------- unit: loopback gate


class TestLoopbackGate:
    def test_loopback_urls_pass(self):
        sld = _discovery()
        assert sld._loopback_url("http://127.0.0.1:1234/v1")
        assert sld._loopback_url("http://localhost:8080")

    def test_remote_urls_rejected(self):
        sld = _discovery()
        assert not sld._loopback_url("http://10.0.0.5:1234/v1")
        assert not sld._loopback_url("https://api.example.com/v1")

    def test_empty_url_fails_closed(self):
        sld = _discovery()
        # base_url_hostname("") -> "" -> not a loopback host -> gate closed
        assert not sld._loopback_url("")


# --------------------------------------------------------------------------- unit: discovery policy


class TestDiscoverLocalModel:
    def test_empty_base_url_is_noop(self):
        sld = _discovery()
        assert sld.discover_local_model("j1", {}, "") == ""

    def test_remote_base_url_never_probed(self):
        sld = _discovery()
        with mock.patch(
            "hermes_cli.runtime_provider._auto_detect_local_model",
            return_value="should-not-be-called",
        ) as probe:
            assert (
                sld.discover_local_model("j1", {"base_url": "http://10.0.0.5:1234/v1"})
                == ""
            )
        probe.assert_not_called()

    def test_reuses_upstream_helper_verbatim(self):
        sld = _discovery()
        with mock.patch(
            "hermes_cli.runtime_provider._auto_detect_local_model",
            return_value="qwen3-4b",
        ) as probe:
            got = sld.discover_local_model(
                "j1", {"base_url": "http://127.0.0.1:1234/v1"}
            )
        assert got == "qwen3-4b"
        probe.assert_called_once_with("http://127.0.0.1:1234/v1")

    def test_provider_default_registry_url_used_when_no_base_url(self):
        sld = _discovery()
        registry_entry = SimpleNamespace(inference_base_url="http://127.0.0.1:1234/v1")
        with (
            mock.patch(
                "hermes_cli.auth.PROVIDER_REGISTRY", {"lmstudio": registry_entry}
            ),
            mock.patch(
                "hermes_cli.runtime_provider._auto_detect_local_model",
                return_value="qwen3-4b",
            ) as probe,
        ):
            got = sld.discover_local_model(
                "j1", {"provider": "lmstudio"}, provider_hint="lmstudio"
            )
        assert got == "qwen3-4b"
        probe.assert_called_once_with("http://127.0.0.1:1234/v1")

    def test_non_local_provider_without_base_url_is_noop(self):
        sld = _discovery()
        with (
            mock.patch("hermes_cli.auth.PROVIDER_REGISTRY", {}),
            mock.patch("hermes_cli.runtime_provider._auto_detect_local_model") as probe,
        ):
            assert sld.discover_local_model("j1", {"provider": "openrouter"}) == ""
        probe.assert_not_called()

    def test_junk_model_cfg_does_not_raise(self):
        sld = _discovery()
        assert sld.discover_local_model("j1", "not-a-dict") == ""
        assert sld.discover_local_model("j1", None) == ""

    def test_probe_failure_absorbed(self):
        sld = _discovery()
        with mock.patch(
            "hermes_cli.runtime_provider._auto_detect_local_model",
            side_effect=RuntimeError("boom"),
        ):
            assert (
                sld.discover_local_model("j1", {"base_url": "http://127.0.0.1:1234/v1"})
                == ""
            )

    def test_explicit_base_url_beats_config(self):
        sld = _discovery()
        with mock.patch(
            "hermes_cli.runtime_provider._auto_detect_local_model", return_value="m"
        ) as probe:
            sld.discover_local_model(
                "j1",
                {"base_url": "http://localhost:9999/v1"},
                explicit_base_url="http://localhost:7777/v1",
            )
        probe.assert_called_once_with("http://localhost:7777/v1")


# --------------------------------------------------------------------------- integration: caller hook


class TestLoadCronJobConfigDiscoveryHook:
    """The hook fires only when the model chain resolved empty; a discovered
    model flows into the run; a silent probe still hits the fail-fast raise."""

    def test_discovery_supplies_model_when_chain_empty(self, monkeypatch):
        from cron import scheduler
        from cron import scheduler_local_discovery as sld

        seen = {}

        def fake_discover(job_id, model_cfg, provider_hint="", explicit_base_url=""):
            seen["args"] = (job_id, model_cfg)
            return "qwen3-4b"

        monkeypatch.setattr(scheduler, "cron_env_setting", lambda name: "")
        monkeypatch.setattr(
            scheduler, "_get_hermes_home", lambda: __import__("pathlib").Path("/tmp")
        )
        monkeypatch.setattr("os.path.exists", lambda path: False)
        monkeypatch.setattr(sld, "discover_local_model", fake_discover)
        jc = scheduler._load_cron_job_config({"model": None}, "job-1", "Test Job")
        assert jc.model == "qwen3-4b"
        assert seen["args"] == ("job-1", {})

    def test_no_discovery_when_job_model_pinned(self, monkeypatch):
        from cron import scheduler
        from cron import scheduler_local_discovery as sld

        def fail_discover(*a, **k):
            raise AssertionError("discovery must not run for pinned jobs")

        monkeypatch.setattr(scheduler, "cron_env_setting", lambda name: "")
        monkeypatch.setattr(
            scheduler, "_get_hermes_home", lambda: __import__("pathlib").Path("/tmp")
        )
        monkeypatch.setattr("os.path.exists", lambda path: False)
        monkeypatch.setattr(sld, "discover_local_model", fail_discover)
        jc = scheduler._load_cron_job_config({"model": "gpt-x"}, "job-1", "Test Job")
        assert jc.model == "gpt-x"

    def test_fail_fast_still_raises_when_discovery_empty(self, monkeypatch):
        from cron import scheduler
        from cron import scheduler_local_discovery as sld

        monkeypatch.setattr(scheduler, "cron_env_setting", lambda name: "")
        monkeypatch.setattr(
            scheduler, "_get_hermes_home", lambda: __import__("pathlib").Path("/tmp")
        )
        monkeypatch.setattr("os.path.exists", lambda path: False)
        monkeypatch.setattr(sld, "discover_local_model", lambda *a, **k: "")
        with pytest.raises(RuntimeError, match="has no model configured"):
            scheduler._load_cron_job_config({"model": None}, "job-1", "Test Job")
