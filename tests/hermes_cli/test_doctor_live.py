"""Tests for ``hermes doctor --live`` — opt-in bounded real-call tool-backend probes.

All probes are mocked at the HTTP/client layer; no real network calls are made.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_cli import doctor_live
from hermes_cli.doctor_live import (
    ProbeResult,
    maybe_run_live_checks,
    run_live_checks,
)
from tools import browser_tool_install as bt_install

# Captured before the autouse fixture below stubs doctor_live._browser_available
# to a constant, so TestBrowserAvailableNpxRung can exercise the real function.
_real_browser_available = doctor_live._browser_available


def _args(live: bool = True) -> argparse.Namespace:
    return argparse.Namespace(live=live)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip backend credentials so each test opts in explicitly."""
    for var in ("FIRECRAWL_API_KEY", "FAL_KEY", "OPENAI_API_KEY",
                "ELEVENLABS_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Default: empty config, no MCP servers, local tts/stt.
    monkeypatch.setattr(doctor_live, "_load_config", lambda: {})
    # Default: browser not installed.
    monkeypatch.setattr(doctor_live, "_browser_available", lambda: False)


class TestLiveFlagGating:
    def test_parser_has_live_flag_default_false(self):
        from hermes_cli.subcommands.doctor import build_doctor_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        build_doctor_parser(sub, cmd_doctor=lambda a: None)
        args = parser.parse_args(["doctor"])
        assert args.live is False
        args = parser.parse_args(["doctor", "--live"])
        assert args.live is True

    def test_no_live_flag_means_zero_probes(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            doctor_live, "run_live_checks",
            lambda *a, **k: called.append(True))
        result = maybe_run_live_checks(_args(live=False), [])
        assert result is None
        assert called == []

    def test_missing_live_attr_means_zero_probes(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            doctor_live, "run_live_checks",
            lambda *a, **k: called.append(True))
        assert maybe_run_live_checks(SimpleNamespace(), []) is None
        assert called == []

    def test_live_flag_runs_checks(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            doctor_live, "run_live_checks",
            lambda issues, **k: called.append(issues) or [])
        issues: list[str] = []
        maybe_run_live_checks(_args(live=True), issues)
        assert called == [issues]

    def test_live_check_crash_never_propagates(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise RuntimeError("probe subsystem exploded")

        monkeypatch.setattr(doctor_live, "run_live_checks", _boom)
        # Must not raise.
        maybe_run_live_checks(_args(live=True), [])


class TestConfiguredOnlySelection:
    def test_all_unconfigured_all_skipped(self, capsys):
        results = run_live_checks([])
        assert results, "expected one result per backend"
        assert all(r.status == "skip" for r in results)
        # No issues appended for skips.

    def test_unconfigured_backends_do_not_touch_network(self, monkeypatch):
        def _no_net(*a, **k):
            raise AssertionError("HTTP call made for unconfigured backend")

        monkeypatch.setattr(doctor_live, "_http_get", _no_net)
        results = run_live_checks([])
        assert all(r.status == "skip" for r in results)

    def test_firecrawl_probed_when_key_present(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        calls = []

        def _fake_get(url, headers=None, timeout=None):
            calls.append(url)
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(doctor_live, "_http_get", _fake_get)
        results = {r.name: r for r in run_live_checks([])}
        assert results["Firecrawl"].status == "pass"
        assert any("firecrawl" in u for u in calls)

    def test_firecrawl_invalid_key_fails_and_appends_issue(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-bad")
        monkeypatch.setattr(
            doctor_live, "_http_get",
            lambda *a, **k: SimpleNamespace(status_code=401))
        issues: list[str] = []
        results = {r.name: r for r in run_live_checks(issues)}
        assert results["Firecrawl"].status == "fail"
        assert any("FIRECRAWL" in i or "Firecrawl" in i for i in issues)

    def test_fal_probed_when_key_present(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "fal-test")
        monkeypatch.setattr(
            doctor_live, "_http_get",
            lambda *a, **k: SimpleNamespace(status_code=200))
        results = {r.name: r for r in run_live_checks([])}
        assert results["FAL"].status == "pass"

    def test_mcp_servers_probed_per_configured_server(self, monkeypatch):
        monkeypatch.setattr(
            doctor_live, "_load_config",
            lambda: {"mcp_servers": {"alpha": {"url": "https://x"},
                                     "beta": {"command": "foo"}}})
        probed = []
        monkeypatch.setattr(
            doctor_live, "_probe_mcp_server",
            lambda name, cfg, timeout: probed.append(name) or [("t", "d")])
        results = [r for r in run_live_checks([]) if r.name.startswith("MCP")]
        assert sorted(probed) == ["alpha", "beta"]
        assert len(results) == 2
        assert all(r.status == "pass" for r in results)

    def test_tts_local_provider_skipped(self, monkeypatch):
        monkeypatch.setattr(
            doctor_live, "_load_config",
            lambda: {"tts": {"provider": "edge"}})
        results = {r.name: r for r in run_live_checks([])}
        assert results["TTS"].status == "skip"

    def test_tts_openai_probed_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            doctor_live, "_load_config",
            lambda: {"tts": {"provider": "openai"}})
        monkeypatch.setattr(
            doctor_live, "_http_get",
            lambda *a, **k: SimpleNamespace(status_code=200))
        results = {r.name: r for r in run_live_checks([])}
        assert results["TTS"].status == "pass"

    def test_stt_groq_probed_with_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setattr(
            doctor_live, "_load_config",
            lambda: {"stt": {"provider": "groq"}})
        monkeypatch.setattr(
            doctor_live, "_http_get",
            lambda *a, **k: SimpleNamespace(status_code=200))
        results = {r.name: r for r in run_live_checks([])}
        assert results["STT"].status == "pass"

    def test_stt_provider_configured_but_key_missing_warns(self, monkeypatch):
        monkeypatch.setattr(
            doctor_live, "_load_config",
            lambda: {"stt": {"provider": "groq"}})
        results = {r.name: r for r in run_live_checks([])}
        assert results["STT"].status == "warn"

    def test_browser_probed_when_available(self, monkeypatch):
        monkeypatch.setattr(doctor_live, "_browser_available", lambda: True)
        monkeypatch.setattr(
            doctor_live, "_launch_browser_probe",
            lambda timeout: (True, "about:blank ok"))
        results = {r.name: r for r in run_live_checks([])}
        assert results["Browser"].status == "pass"


class TestBrowserAvailableNpxRung:
    """agent-browser resolves lazily via npx on the default install (#43564),
    invisible to the bare PATH/node_modules probes _browser_available starts
    with. It must fall through to the same cascade `hermes doctor` uses."""

    def _block_path_and_node_modules_checks(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda *a, **k: None)
        monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", tmp_path / "home")
        monkeypatch.setattr("hermes_cli.doctor.PROJECT_ROOT", tmp_path / "root")

    def test_true_when_npx_resolves_agent_browser(self, monkeypatch, tmp_path):
        self._block_path_and_node_modules_checks(monkeypatch, tmp_path)

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "npx agent-browser")

        assert _real_browser_available() is True

    def test_false_when_nothing_resolves(self, monkeypatch, tmp_path):
        self._block_path_and_node_modules_checks(monkeypatch, tmp_path)

        def _raise(**_kw):
            raise FileNotFoundError("agent-browser CLI not found")

        monkeypatch.setattr(bt_install, "_find_agent_browser", _raise)

        assert _real_browser_available() is False

    def test_false_on_termux_local_bare_npx(self, monkeypatch, tmp_path):
        """On Termux in local mode the bare npx fallback is too fragile to
        advertise as ready — must not diverge from dep_ensure/nous_subscription's
        same carve-out."""
        self._block_path_and_node_modules_checks(monkeypatch, tmp_path)

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "npx agent-browser")
        monkeypatch.setattr("tools.browser_tool_install._requires_real_termux_browser_install", lambda cmd: True)

        assert _real_browser_available() is False


class TestFailureIsolation:
    def test_one_probe_raising_does_not_stop_others(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        monkeypatch.setenv("FAL_KEY", "fal-test")

        def _get(url, headers=None, timeout=None):
            if "firecrawl" in url:
                raise RuntimeError("connection reset")
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(doctor_live, "_http_get", _get)
        issues: list[str] = []
        results = {r.name: r for r in run_live_checks(issues)}
        assert results["Firecrawl"].status == "fail"
        assert results["FAL"].status == "pass"

    def test_mcp_probe_failure_isolated_per_server(self, monkeypatch):
        monkeypatch.setattr(
            doctor_live, "_load_config",
            lambda: {"mcp_servers": {"bad": {"url": "https://x"},
                                     "good": {"url": "https://y"}}})

        def _probe(name, cfg, timeout):
            if name == "bad":
                raise ConnectionError("refused")
            return [("tool", "desc")]

        monkeypatch.setattr(doctor_live, "_probe_mcp_server", _probe)
        results = {r.name: r for r in run_live_checks([])}
        assert results["MCP: bad"].status == "fail"
        assert results["MCP: good"].status == "pass"


class TestTimeoutHandling:
    def test_timeout_reported_as_fail(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")

        def _slow(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr(doctor_live, "_http_get", _slow)
        results = {r.name: r for r in run_live_checks([])}
        assert results["Firecrawl"].status == "fail"
        assert "time" in (results["Firecrawl"].detail or "").lower()

    def test_probe_timeout_bounded_and_configurable(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        monkeypatch.setattr(
            doctor_live, "_load_config",
            lambda: {"doctor": {"live_probe_timeout": 3}})
        seen = {}

        def _get(url, headers=None, timeout=None):
            seen["timeout"] = timeout
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(doctor_live, "_http_get", _get)
        run_live_checks([])
        assert seen["timeout"] == 3

    def test_default_timeout_is_10s(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        seen = {}

        def _get(url, headers=None, timeout=None):
            seen["timeout"] = timeout
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(doctor_live, "_http_get", _get)
        run_live_checks([])
        assert seen["timeout"] == 10.0


class TestReadOnly:
    def test_probe_result_is_plain_record(self):
        r = ProbeResult(name="X", status="skip", detail="not configured")
        assert (r.name, r.status, r.detail) == ("X", "skip", "not configured")

    def test_skips_never_append_issues(self, capsys):
        issues: list[str] = []
        run_live_checks(issues)
        assert issues == []
