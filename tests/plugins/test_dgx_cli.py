"""Tests for plugins/dgx/cli.py

Covers the shipped functionality: HTTP/SSH helpers, probe functions, argparse
wiring, dispatch, and the setup/status/models/use/endpoint/pull/rm/ps/push/
doctor/watch/formation/nim/node commands. All tests are live (not xfail).

Note: `hermes dgx run` (arbitrary remote shell) and the `dgx_run` agent tool
were intentionally removed — free-form remote execution goes through the host
terminal tool, which is gated by the dangerous-command approval system.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dgx_defaults():
    from plugins.dgx._dgx_config import DEFAULTS
    d = dict(DEFAULTS)
    # Tests assume a configured DGX. The shipped DEFAULTS leave host=None
    # so that an un-configured install never accidentally talks to a stranger.
    d.update({
        "host": "10.0.0.1",
        "ssh_user": "dgx",
        "ollama_port": 11434,
        "vllm_port": 30800,
        "vllm_32b_port": 30881,
        "litellm_host": "10.0.0.2",
        "litellm_port": 4000,
    })
    return d


@pytest.fixture
def mock_config(monkeypatch, dgx_defaults):
    """Patch load_dgx_config and apply_endpoint to avoid real file I/O."""
    stored = {"dgx": dict(dgx_defaults), "model": {}}

    def _load_dgx():
        return dict(dgx_defaults)

    def _load():
        return dict(stored)

    def _save(cfg):
        stored.clear()
        stored.update(cfg)

    import plugins.dgx.cli as cli_mod
    import plugins.dgx._dgx_config as cfg_mod

    monkeypatch.setattr(cli_mod, "load_dgx_config", _load_dgx)
    monkeypatch.setattr(cfg_mod, "load_config", _load, raising=False)
    monkeypatch.setattr(cfg_mod, "save_config", _save, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", _load)
    monkeypatch.setattr("hermes_cli.config.save_config", _save)
    return stored


# ---------------------------------------------------------------------------
# _get_json
# ---------------------------------------------------------------------------

class TestGetJson:
    def test_returns_parsed_json_on_success(self):
        import urllib.request
        from plugins.dgx.cli import _get_json
        payload = json.dumps({"models": []}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            data, err = _get_json("http://localhost:11434/api/tags")
        assert data == {"models": []}
        assert err is None

    def test_returns_none_and_error_on_connection_refused(self):
        import urllib.error
        from plugins.dgx.cli import _get_json
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            data, err = _get_json("http://localhost:11434/api/tags")
        assert data is None
        assert err is not None

    def test_returns_none_on_timeout(self):
        import urllib.error
        from plugins.dgx.cli import _get_json
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            data, err = _get_json("http://localhost:11434/api/tags", timeout=1)
        assert data is None


# ---------------------------------------------------------------------------
# _check_endpoint
# ---------------------------------------------------------------------------

class TestCheckEndpoint:
    def test_returns_true_when_reachable(self):
        from plugins.dgx.cli import _check_endpoint
        with patch("plugins.dgx.cli._get_json", return_value=({"ok": True}, None)):
            ok, msg = _check_endpoint("http://localhost:4000/health")
        assert ok is True

    def test_returns_false_when_not_reachable(self):
        from plugins.dgx.cli import _check_endpoint
        with patch("plugins.dgx.cli._get_json", return_value=(None, "Connection refused")):
            ok, msg = _check_endpoint("http://localhost:4000/health")
        assert ok is False
        assert "Connection refused" in msg


# ---------------------------------------------------------------------------
# _ssh_run
# ---------------------------------------------------------------------------

class TestSshRun:
    def test_returns_stdout_on_success(self):
        from plugins.dgx.cli import _ssh_run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU 0, A100, 10000, 40960, 25\n"
        with patch("subprocess.run", return_value=mock_result):
            ok, out = _ssh_run("dgx", "10.0.0.1", "nvidia-smi")
        assert ok is True
        assert "GPU 0" in out

    def test_returns_false_on_nonzero_exit(self):
        from plugins.dgx.cli import _ssh_run
        mock_result = MagicMock()
        mock_result.returncode = 255
        mock_result.stdout = ""
        mock_result.stderr = "Connection refused"
        with patch("subprocess.run", return_value=mock_result):
            ok, out = _ssh_run("dgx", "10.0.0.1", "nvidia-smi")
        assert ok is False

    def test_returns_false_on_timeout(self):
        from plugins.dgx.cli import _ssh_run
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 10)):
            ok, out = _ssh_run("dgx", "10.0.0.1", "nvidia-smi")
        assert ok is False
        assert "timed out" in out

    def test_returns_false_when_ssh_not_found(self):
        from plugins.dgx.cli import _ssh_run
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ok, out = _ssh_run("dgx", "10.0.0.1", "nvidia-smi")
        assert ok is False
        assert "not found" in out


# ---------------------------------------------------------------------------
# _probe_ollama / _probe_vllm / _probe_litellm
# ---------------------------------------------------------------------------

class TestProbes:
    def test_probe_ollama_returns_model_names_on_success(self):
        from plugins.dgx.cli import _probe_ollama
        payload = {"models": [{"name": "nemotron3:33b"}, {"name": "qwen2.5-coder:14b"}]}
        with patch("plugins.dgx.cli._get_json", return_value=(payload, None)):
            ok, models = _probe_ollama("10.0.0.1", 11434)
        assert ok is True
        assert "nemotron3:33b" in models
        assert len(models) == 2

    def test_probe_ollama_returns_false_on_failure(self):
        from plugins.dgx.cli import _probe_ollama
        with patch("plugins.dgx.cli._get_json", return_value=(None, "refused")):
            ok, models = _probe_ollama("10.0.0.1", 11434)
        assert ok is False
        assert models == []

    def test_probe_vllm_returns_model_ids_on_success(self):
        from plugins.dgx.cli import _probe_vllm
        payload = {"data": [{"id": "qwen2.5-coder-3b"}]}
        with patch("plugins.dgx.cli._get_json", return_value=(payload, None)):
            ok, models = _probe_vllm("10.0.0.1", 30800)
        assert ok is True
        assert "qwen2.5-coder-3b" in models

    def test_probe_vllm_returns_false_on_failure(self):
        from plugins.dgx.cli import _probe_vllm
        with patch("plugins.dgx.cli._get_json", return_value=(None, "refused")):
            ok, models = _probe_vllm("10.0.0.1", 30800)
        assert ok is False

    def test_probe_litellm_returns_true_when_healthy(self):
        from plugins.dgx.cli import _probe_litellm
        with patch("plugins.dgx.cli._check_endpoint", return_value=(True, "ok")):
            assert _probe_litellm("10.0.0.2", 4000) is True

    def test_probe_litellm_returns_false_when_down(self):
        from plugins.dgx.cli import _probe_litellm
        with patch("plugins.dgx.cli._check_endpoint", return_value=(False, "refused")):
            assert _probe_litellm("10.0.0.2", 4000) is False


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

class TestArgparseWiring:
    def _parser(self):
        from plugins.dgx.cli import register_cli
        p = argparse.ArgumentParser(prog="hermes dgx")
        register_cli(p)
        return p

    def test_no_subcommand_dispatches_to_dgx_command(self):
        from plugins.dgx.cli import dgx_command
        p = self._parser()
        ns = p.parse_args([])
        assert ns.func is dgx_command

    def test_setup_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["setup"])
        assert ns.dgx_command == "setup"

    def test_status_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["status"])
        assert ns.dgx_command == "status"

    def test_use_subcommand_parses_model_arg(self):
        p = self._parser()
        ns = p.parse_args(["use", "nemotron3:33b"])
        assert ns.dgx_command == "use"
        assert ns.model == "nemotron3:33b"

    def test_use_subcommand_parses_endpoint_flag(self):
        p = self._parser()
        ns = p.parse_args(["use", "qwen2.5-coder:14b", "--endpoint", "vllm"])
        assert ns.endpoint == "vllm"

    def test_endpoint_subcommand_parses_name(self):
        p = self._parser()
        for ep in ("ollama", "vllm", "litellm"):
            ns = p.parse_args(["endpoint", ep])
            assert ns.name == ep

    def test_endpoint_rejects_unknown_name(self):
        p = self._parser()
        with pytest.raises(SystemExit):
            p.parse_args(["endpoint", "bogus"])

    # --- Tier 1: implemented ---

    def test_pull_subcommand_parses_model_arg(self):
        p = self._parser()
        ns = p.parse_args(["pull", "nemotron3:70b"])
        assert ns.dgx_command == "pull"
        assert ns.model == "nemotron3:70b"

    def test_rm_subcommand_parses_model_arg(self):
        p = self._parser()
        ns = p.parse_args(["rm", "old-model:latest"])
        assert ns.dgx_command == "rm"

    def test_rm_force_flag(self):
        p = self._parser()
        ns = p.parse_args(["rm", "old-model:latest", "--force"])
        assert ns.force is True

    def test_ps_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["ps"])
        assert ns.dgx_command == "ps"

    # --- Tier 2: implemented ---

    def test_run_subcommand_removed(self):
        # Regression: the `dgx run` arbitrary-command subcommand was removed.
        # Free-form remote shell belongs to the host terminal tool (gated by
        # the dangerous-command approval system), not a bespoke unguarded
        # SSH-exec path. argparse must now reject `run` as an unknown command.
        p = self._parser()
        with pytest.raises(SystemExit):
            p.parse_args(["run", "nvidia-smi"])

    def test_push_subcommand_parses_local_path(self):
        p = self._parser()
        ns = p.parse_args(["push", "./myproject"])
        assert ns.dgx_command == "push"
        assert ns.local == "./myproject"
        assert ns.remote is None

    def test_push_subcommand_parses_remote_path(self):
        p = self._parser()
        ns = p.parse_args(["push", "./myproject", "~/code/"])
        assert ns.remote == "~/code/"

    def test_doctor_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["doctor"])
        assert ns.dgx_command == "doctor"

    def test_watch_subcommand_parses(self):
        p = self._parser()
        ns = p.parse_args(["watch"])
        assert ns.dgx_command == "watch"

    def test_watch_interval_flag(self):
        p = self._parser()
        ns = p.parse_args(["watch", "--interval", "5"])
        assert ns.interval == 5


# ---------------------------------------------------------------------------
# dgx_command dispatch
# ---------------------------------------------------------------------------

class TestDgxCommandDispatch:
    def test_no_subcommand_prints_usage_and_returns_2(self, capsys):
        from plugins.dgx.cli import dgx_command
        ret = dgx_command(SimpleNamespace(dgx_command=None))
        assert ret == 2
        assert "usage" in capsys.readouterr().out.lower()

    def test_unknown_subcommand_returns_2(self, capsys):
        from plugins.dgx.cli import dgx_command
        ret = dgx_command(SimpleNamespace(dgx_command="bogus"))
        assert ret == 2


# ---------------------------------------------------------------------------
# Unconfigured-DGX dispatch (C5)
# ---------------------------------------------------------------------------

class TestUnconfiguredDispatch:
    """Regression (C5): on an unconfigured install (host=None), commands that
    call the URL helpers must surface the "run hermes dgx setup" guidance and
    exit 1 — NOT raise an uncaught DGXNotConfigured traceback. dgx_command
    catches it at one shared point.
    """

    @pytest.fixture
    def unconfigured(self, monkeypatch):
        from plugins.dgx._dgx_config import DEFAULTS
        d = dict(DEFAULTS)  # host is None — the unconfigured default
        monkeypatch.setattr("plugins.dgx.cli.load_dgx_config", lambda: dict(d))
        # Never spawn real ssh / http; the point under test is the raise path.
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "no host"))
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda *a, **k: (None, "no host"))

    def test_status_unconfigured_returns_1_with_setup_hint(self, unconfigured, capsys):
        from plugins.dgx.cli import dgx_command
        ret = dgx_command(SimpleNamespace(dgx_command="status"))
        out = capsys.readouterr().out
        assert ret == 1
        assert "hermes dgx setup" in out

    def test_doctor_unconfigured_returns_1_with_setup_hint(self, unconfigured, capsys):
        from plugins.dgx.cli import dgx_command
        ret = dgx_command(SimpleNamespace(dgx_command="doctor"))
        out = capsys.readouterr().out
        assert ret == 1
        assert "hermes dgx setup" in out


# ---------------------------------------------------------------------------
# _cmd_status: GPU [N/A] handling
# ---------------------------------------------------------------------------

class TestStatusNvidiaNA:
    """Regression: DGX Spark (aarch64/GB10) returns [N/A] for some smi fields."""

    def test_status_handles_na_memory_fields_without_crash(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_status

        na_line = "[N/A], NVIDIA GH200 120GB, [N/A], 98304, [N/A]"
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, na_line))
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda url, **k: ({"models": [], "data": []}, None))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "unreachable"))

        ret = _cmd_status()
        out = capsys.readouterr().out
        assert "GH200" in out
        assert ret == 0

    def test_status_renders_bar_for_numeric_fields(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_status

        line = "0, NVIDIA A100, 20480, 40960, 50"
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, line))
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda url, **k: ({"models": [], "data": []}, None))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "unreachable"))

        _cmd_status()
        out = capsys.readouterr().out
        assert "█" in out
        assert "20480/40960" in out


# ---------------------------------------------------------------------------
# _cmd_endpoint
# ---------------------------------------------------------------------------

class TestCmdEndpoint:
    # _cmd_endpoint calls the `apply_endpoint` name imported INTO
    # plugins.dgx.cli, so the mock must patch the cli namespace. Patching
    # plugins.dgx._dgx_config.apply_endpoint is a no-op (it doesn't rebind the
    # already-imported name) — the prior tests did that and silently ran the
    # real apply_endpoint. Assert the call to lock the isolation in.
    def test_switches_to_ollama(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_endpoint
        import plugins.dgx.cli as cli_mod
        calls = []
        monkeypatch.setattr(cli_mod, "apply_endpoint",
                            lambda dgx, ep=None, **k: calls.append(ep))
        ret = _cmd_endpoint("ollama")
        assert ret == 0
        assert calls == ["ollama"]
        assert "ollama" in capsys.readouterr().out.lower()

    def test_switches_to_vllm(self, mock_config, capsys, monkeypatch):
        from plugins.dgx.cli import _cmd_endpoint
        import plugins.dgx.cli as cli_mod
        calls = []
        monkeypatch.setattr(cli_mod, "apply_endpoint",
                            lambda dgx, ep=None, **k: calls.append(ep))
        ret = _cmd_endpoint("vllm")
        assert ret == 0
        assert calls == ["vllm"]


# ---------------------------------------------------------------------------
# Tier 1 feature tests (red — implement these next)
# ---------------------------------------------------------------------------

class TestTier1Pull:
    def test_pull_runs_ollama_pull_via_ssh(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_pull
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_stream", lambda u, h, cmd, **k: calls.append(cmd) or 0)
        _cmd_pull("nemotron3:70b")
        assert any("ollama pull nemotron3:70b" in c for c in calls)

    def test_pull_returns_nonzero_on_ssh_failure(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_pull
        monkeypatch.setattr("plugins.dgx.cli._ssh_stream", lambda *a, **k: 1)
        ret = _cmd_pull("some-model:latest")
        assert ret != 0

    def test_pull_prints_host_before_streaming(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_pull
        monkeypatch.setattr("plugins.dgx.cli._ssh_stream", lambda *a, **k: 0)
        _cmd_pull("gemma4:26b")
        out = capsys.readouterr().out
        assert "gemma4:26b" in out


class TestTier1Rm:
    def test_rm_runs_ollama_rm_via_ssh(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_rm
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda u, h, cmd, **k: calls.append(cmd) or (True, ""))
        monkeypatch.setattr("builtins.input", lambda _: "y")
        _cmd_rm("old-model:latest")
        assert any("ollama rm old-model:latest" in c for c in calls)

    def test_rm_aborts_on_no_confirmation(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_rm
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: calls.append(True) or (True, ""))
        monkeypatch.setattr("builtins.input", lambda _: "n")
        _cmd_rm("old-model:latest")
        assert len(calls) == 0

    def test_rm_force_skips_prompt(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_rm
        calls = []
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda u, h, cmd, **k: calls.append(cmd) or (True, ""))
        _cmd_rm("old-model:latest", force=True)
        assert any("ollama rm" in c for c in calls)

    def test_rm_returns_nonzero_on_ssh_failure(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_rm
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "error"))
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert _cmd_rm("bad-model:latest") != 0


class TestTier1Ps:
    def test_ps_shows_loaded_models(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_ps
        ollama_ps_output = "NAME\t\tID\t\tSIZE\tPROCESSOR\nnemotron3:33b\tabc123\t20 GB\tgpu"
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, ollama_ps_output))
        ret = _cmd_ps()
        out = capsys.readouterr().out
        assert "nemotron3:33b" in out
        assert ret == 0

    def test_ps_graceful_when_nothing_loaded(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_ps
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (True, "NAME\tID\tSIZE\tPROCESSOR"))
        ret = _cmd_ps()
        assert ret == 0

    def test_ps_returns_1_when_ssh_fails(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_ps
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "refused"))
        assert _cmd_ps() == 1


# ---------------------------------------------------------------------------
# Tier 2 feature tests (red)
# ---------------------------------------------------------------------------

class TestTier2Push:
    def test_push_calls_rsync(self, mock_config, monkeypatch, tmp_path):
        from plugins.dgx.cli import _cmd_push
        calls = []
        monkeypatch.setattr("subprocess.run",
                            lambda cmd, **k: calls.append(cmd) or MagicMock(returncode=0))
        (tmp_path / "file.py").write_text("x = 1")
        ret = _cmd_push(str(tmp_path / "file.py"), None)
        assert any("rsync" in str(c) for c in calls)
        assert ret == 0

    def test_push_uses_default_remote_path(self, mock_config, monkeypatch, tmp_path):
        from plugins.dgx.cli import _cmd_push
        calls = []
        monkeypatch.setattr("subprocess.run",
                            lambda cmd, **k: calls.append(cmd) or MagicMock(returncode=0))
        _cmd_push("/some/local/path", None)
        assert any("~/workspace/" in str(c) for c in calls)

    def test_push_uses_specified_remote_path(self, mock_config, monkeypatch):
        from plugins.dgx.cli import _cmd_push
        calls = []
        monkeypatch.setattr("subprocess.run",
                            lambda cmd, **k: calls.append(cmd) or MagicMock(returncode=0))
        _cmd_push("/local/path", "~/code/myproject/")
        assert any("~/code/myproject/" in str(c) for c in calls)

    def test_push_returns_1_when_rsync_not_found(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_push
        monkeypatch.setattr("subprocess.run", MagicMock(side_effect=FileNotFoundError))
        ret = _cmd_push("/some/path", None)
        assert ret == 1
        assert "rsync not found" in capsys.readouterr().out


class TestTier2Doctor:
    def _all_ok(self):
        """Mock SSH returning ok for all calls."""
        def _ssh(u, h, cmd, **k):
            if cmd == "echo ok":
                return (True, "ok")
            # nvidia-smi call
            return (True, "0, NVIDIA GH200, 20480, 98304, 25")
        return _ssh

    def test_doctor_reports_all_checks(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_doctor
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", self._all_ok())
        monkeypatch.setattr("plugins.dgx.cli._get_json",
                            lambda url, **k: ({"models": [], "data": []}, None))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (True, "ok"))
        ret = _cmd_doctor()
        out = capsys.readouterr().out
        for check in ("ssh", "ollama", "vllm", "gpu"):
            assert check.lower() in out.lower(), f"missing check: {check}"
        assert ret == 0

    def test_doctor_returns_nonzero_when_ssh_unreachable(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_doctor
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", lambda *a, **k: (False, "refused"))
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda *a, **k: (None, "refused"))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "refused"))
        ret = _cmd_doctor()
        assert ret != 0

    def test_doctor_returns_nonzero_when_all_inference_down(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_doctor
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", self._all_ok())
        monkeypatch.setattr("plugins.dgx.cli._get_json", lambda *a, **k: (None, "refused"))
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "ok"))
        ret = _cmd_doctor()
        assert ret != 0

    def test_doctor_passes_when_litellm_down_but_ollama_ok(self, mock_config, monkeypatch, capsys):
        from plugins.dgx.cli import _cmd_doctor
        monkeypatch.setattr("plugins.dgx.cli._ssh_run", self._all_ok())
        def _get(url, **k):
            if "api/tags" in url:
                return ({"models": []}, None)
            if "v1/models" in url:
                return (None, "refused")   # vLLM down
            return (None, "refused")
        monkeypatch.setattr("plugins.dgx.cli._get_json", _get)
        monkeypatch.setattr("plugins.dgx.cli._check_endpoint", lambda *a, **k: (False, "key required"))
        ret = _cmd_doctor()
        # Ollama up → inference_ok → passes
        assert ret == 0


class TestTier2Watch:
    def test_watch_exits_cleanly_on_keyboard_interrupt(self, mock_config, monkeypatch):
        import time
        from plugins.dgx.cli import _cmd_watch

        call_count = [0]

        def _fake_ssh_run(*a, **k):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt
            return (True, "0, A100, 20480, 40960, 50")

        monkeypatch.setattr("plugins.dgx.cli._ssh_run", _fake_ssh_run)
        monkeypatch.setattr("time.sleep", lambda s: None)
        ret = _cmd_watch(interval=0)
        assert ret == 0

    def test_watch_handles_ssh_failure_gracefully(self, mock_config, monkeypatch, capsys):
        import time
        from plugins.dgx.cli import _cmd_watch

        call_count = [0]

        def _fail_then_interrupt(*a, **k):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt
            return (False, "connection refused")

        monkeypatch.setattr("plugins.dgx.cli._ssh_run", _fail_then_interrupt)
        monkeypatch.setattr("time.sleep", lambda s: None)
        ret = _cmd_watch(interval=0)
        assert ret == 0


# ---------------------------------------------------------------------------
# Tier 3 feature tests (red)
# ---------------------------------------------------------------------------

class TestTier3AgentTools:
    def _registered_tools(self):
        from plugins.dgx import register
        tools = []

        class FakeCtx:
            def register_cli_command(self, **k): pass
            def register_tool(self, name, **k): tools.append(name)

        register(FakeCtx())
        return tools

    def _registered_tool_kwargs(self):
        from plugins.dgx import register
        captured = {}

        class FakeCtx:
            def register_cli_command(self, **k): pass
            def register_tool(self, name, **k): captured[name] = k

        register(FakeCtx())
        return captured

    def test_dgx_gpu_status_tool_registered(self):
        assert "dgx_gpu_status" in self._registered_tools()

    def test_dgx_run_tool_not_registered(self):
        # Regression: dgx_run was an unguarded arbitrary-RCE agent tool — a
        # model could run any shell command on the DGX over SSH, bypassing the
        # host's dangerous-command approval gate. It must NOT be registered.
        assert "dgx_run" not in self._registered_tools()

    def test_dgx_pull_model_tool_registered(self):
        assert "dgx_pull_model" in self._registered_tools()

    def test_handle_dgx_run_removed(self):
        # The handler is gone too — importing it must fail.
        import plugins.dgx.tools as dgx_tools
        assert not hasattr(dgx_tools, "handle_dgx_run")
        assert not hasattr(dgx_tools, "DGX_RUN_SCHEMA")

    def test_agent_tools_gated_by_check_fn(self):
        # C3: each agent tool must carry a check_fn so an enabled-but-
        # unconfigured plugin doesn't expose it (with host=None) to the model.
        kw = self._registered_tool_kwargs()
        for name in ("dgx_gpu_status", "dgx_pull_model"):
            assert kw[name].get("check_fn") is not None, f"{name} has no check_fn"

    def test_check_fn_false_when_unconfigured(self, monkeypatch):
        from plugins.dgx import _dgx_configured
        from plugins.dgx._dgx_config import DEFAULTS
        monkeypatch.setattr("plugins.dgx._dgx_config.load_dgx_config",
                            lambda: dict(DEFAULTS))  # host=None
        assert _dgx_configured() is False

    def test_check_fn_true_when_configured(self, monkeypatch):
        from plugins.dgx import _dgx_configured
        from plugins.dgx._dgx_config import DEFAULTS
        d = dict(DEFAULTS); d["host"] = "10.0.0.1"
        monkeypatch.setattr("plugins.dgx._dgx_config.load_dgx_config", lambda: dict(d))
        assert _dgx_configured() is True

    def test_handle_dgx_gpu_status_unconfigured_returns_hint(self, monkeypatch):
        from plugins.dgx.tools import handle_dgx_gpu_status
        from plugins.dgx._dgx_config import DEFAULTS
        monkeypatch.setattr("plugins.dgx._dgx_config.load_dgx_config",
                            lambda: dict(DEFAULTS))  # host=None
        out = handle_dgx_gpu_status()
        assert "hermes dgx setup" in out

    def test_handle_dgx_pull_model_unconfigured_returns_hint(self, monkeypatch):
        from plugins.dgx.tools import handle_dgx_pull_model
        from plugins.dgx._dgx_config import DEFAULTS
        monkeypatch.setattr("plugins.dgx._dgx_config.load_dgx_config",
                            lambda: dict(DEFAULTS))  # host=None
        out = handle_dgx_pull_model(model="foo:latest")
        assert "hermes dgx setup" in out

    def test_handle_dgx_pull_model_success(self, mock_config, monkeypatch):
        from plugins.dgx.tools import handle_dgx_pull_model
        monkeypatch.setattr("plugins.dgx.cli._ssh_run",
                            lambda *a, **k: (True, ""))
        out = handle_dgx_pull_model(model="nemotron3:33b")
        assert "successfully" in out.lower()

    def test_handle_dgx_pull_model_failure(self, mock_config, monkeypatch):
        from plugins.dgx.tools import handle_dgx_pull_model
        monkeypatch.setattr("plugins.dgx.cli._ssh_run",
                            lambda *a, **k: (False, "no space left"))
        out = handle_dgx_pull_model(model="huge-model:latest")
        assert "failed" in out.lower()

    def test_handle_dgx_gpu_status_includes_gpu_line(self, mock_config, monkeypatch):
        from plugins.dgx.tools import handle_dgx_gpu_status
        monkeypatch.setattr("plugins.dgx.cli._ssh_run",
                            lambda u, h, cmd, **k: (True, "0, A100, 20480, 40960, 50")
                            if "nvidia-smi" in cmd else (True, "model:latest"))
        out = handle_dgx_gpu_status()
        # Assert the actual GPU data line is present, not just the constant
        # "GPU" label (which the failure branch also emits) — see the test-
        # quality finding: `assert "GPU" in out` passed even when SSH failed.
        assert "A100" in out
        assert "40960" in out
