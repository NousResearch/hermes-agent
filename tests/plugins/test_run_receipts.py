"""Tests for the run-receipts plugin.

Covers the bundled plugin at ``plugins/run-receipts/``:

  * ``receipts`` library: pending-run accumulation, receipt finalization,
    hash-chain write, ``verify_chain`` tamper detection.
  * Plugin ``__init__``: hook callbacks feed the accumulator; the
    ``on_session_end`` hook writes one chained receipt per run.
  * Slash command + CLI handler: latest / stats / verify.
  * Privacy: raw args, results, and error messages never reach disk.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME per test (mirrors the disk-cleanup test fixture)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "run-receipts"


def _load_lib():
    spec = importlib.util.spec_from_file_location(
        "run_receipts_under_test", _PLUGIN_DIR / "receipts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def lib():
    return _load_lib()


@pytest.fixture()
def plugin(lib, monkeypatch):
    """Load the plugin package and register it against a stub ctx.

    Returns (module, ctx_stubs) where ctx_stubs carries the registered hooks,
    slash command, and cli command so tests can drive the real wiring.
    """
    pkg_name = "hermes_plugins.run_receipts_test"
    if "hermes_plugins" not in sys.modules:
        import types

        parent = types.ModuleType("hermes_plugins")
        parent.__path__ = []
        sys.modules["hermes_plugins"] = parent
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    # Bind the sibling module under the package name so `from . import receipts` resolves.
    sys.modules[f"{pkg_name}.receipts"] = lib
    spec.loader.exec_module(mod)

    hooks = {}
    commands = {}
    cli_commands = {}

    class _Ctx:
        def register_hook(self, name, callback):
            hooks[name] = callback

        def register_command(self, name, handler=None, description="", **kwargs):
            commands[name] = handler

        def register_cli_command(
            self, name, help="", setup_fn=None, handler_fn=None, description=""
        ):
            cli_commands[name] = (setup_fn, handler_fn)

    ctx = _Ctx()
    mod.register(ctx)
    return mod, {"hooks": hooks, "commands": commands, "cli": cli_commands}


def _run_one_turn(plugin_stubs, session_id="sess-1"):
    """Drive the registered hooks through one realistic turn."""
    hooks = plugin_stubs["hooks"]
    hooks["on_session_start"](session_id=session_id, model="test-model", platform="cli")
    hooks["post_api_request"](
        session_id=session_id,
        task_id="task-1",
        turn_id="turn-1",
        api_request_id="req-1",
        model="test-model",
        provider="test-provider",
        api_duration=0.5,
        api_call_count=1,
    )
    hooks["post_tool_call"](
        session_id=session_id,
        task_id="task-1",
        tool_call_id="call-1",
        tool_name="terminal",
        args={"command": "SECRET_COMMAND xyz"},
        result="SECRET_RESULT xyz",
        duration_ms=12,
    )
    hooks["on_session_end"](
        session_id=session_id,
        task_id="task-1",
        turn_id="turn-1",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
        model="test-model",
        platform="cli",
    )


def _receipts_file(tmp_path):
    return tmp_path / ".hermes" / "receipts" / "runs.ndjson"


class TestRegistration:
    def test_registers_hooks_and_commands(self, plugin):
        _mod, stubs = plugin
        assert set(stubs["hooks"]) == {
            "on_session_start",
            "post_tool_call",
            "post_api_request",
            "api_request_error",
            "on_session_end",
            "on_session_finalize",
        }
        assert "receipts" in stubs["commands"]
        assert "receipts" in stubs["cli"]


class TestReceiptWrite:
    def test_one_turn_writes_one_receipt(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs)
        path = _receipts_file(tmp_path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["v"] == 1
        assert rec["session_id"] == "sess-1"
        assert rec["turn_id"] == "turn-1"
        assert rec["outcome"] == "completed"
        assert rec["exit_reason"] == "text_response(stop)"
        assert rec["model"] == "test-model"
        assert rec["platform"] == "cli"
        assert rec["counts"] == {"tool_calls": 1, "api_calls": 1, "errors": 0}
        assert len(rec["sha256"]) == 64
        assert len(rec["prev_sha256"]) == 64

    def test_tool_call_records_digests_not_content(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs)
        raw = _receipts_file(tmp_path).read_text()
        assert "SECRET_COMMAND" not in raw
        assert "SECRET_RESULT" not in raw
        rec = json.loads(raw.strip())
        tc = rec["tool_calls"][0]
        assert tc["name"] == "terminal"
        assert tc["args_sha256"] != ""
        assert tc["result_sha256"] != ""
        assert tc["duration_ms"] == 12

    def test_api_error_records_digest_and_metadata(self, plugin, tmp_path):
        _mod, stubs = plugin
        hooks = stubs["hooks"]
        hooks["api_request_error"](
            session_id="sess-2",
            task_id="t",
            turn_id="turn-9",
            api_request_id="req-bad",
            model="m",
            provider="p",
            api_duration=1.0,
            api_call_count=3,
            status_code=500,
            retryable=True,
            reason="server_error",
            error={"type": "server_error", "message": "SECRET_BODY"},
        )
        hooks["on_session_end"](
            session_id="sess-2",
            completed=False,
            failed=True,
            interrupted=False,
            turn_exit_reason="max_retries",
            model="m",
            platform="cli",
        )
        raw = _receipts_file(tmp_path).read_text()
        assert "SECRET_BODY" not in raw
        rec = json.loads(raw.strip())
        assert rec["outcome"] == "failed"
        assert rec["counts"]["errors"] == 1
        call = rec["api_calls"][0]
        assert call["ok"] is False
        assert call["status_code"] == 500
        assert call["reason"] == "server_error"

    def test_interrupted_outcome(self, plugin, tmp_path):
        _mod, stubs = plugin
        stubs["hooks"]["on_session_end"](
            session_id="sess-x",
            interrupted=True,
            turn_exit_reason="interrupt",
        )
        rec = json.loads(_receipts_file(tmp_path).read_text().strip())
        assert rec["outcome"] == "interrupted"

    def test_chain_links_across_runs(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs, session_id="s1")
        _run_one_turn(stubs, session_id="s2")
        lines = _receipts_file(tmp_path).read_text().strip().split("\n")
        r1, r2 = (json.loads(l) for l in lines)
        assert r2["prev_sha256"] == r1["sha256"]


class TestVerify:
    def test_verify_intact_chain(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs, "a")
        _run_one_turn(stubs, "b")
        lib = _load_lib()
        report = lib.verify_chain(_receipts_file(tmp_path))
        assert report["ok"] is True
        assert report["records"] == 2

    def test_verify_detects_edited_record(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs, "a")
        path = _receipts_file(tmp_path)
        rec = json.loads(path.read_text().strip())
        rec["outcome"] = "failed"  # tamper
        path.write_text(json.dumps(rec, sort_keys=True) + "\n")
        lib = _load_lib()
        report = lib.verify_chain(path)
        assert report["ok"] is False
        assert "sha256 mismatch" in report["errors"][0]

    def test_verify_detects_deleted_line(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs, "a")
        _run_one_turn(stubs, "b")
        path = _receipts_file(tmp_path)
        lines = path.read_text().strip().split("\n")
        path.write_text(lines[1] + "\n")  # drop first receipt
        lib = _load_lib()
        report = lib.verify_chain(path)
        assert report["ok"] is False
        assert "prev_sha256" in report["errors"][0]

    def test_verify_missing_file(self, lib, tmp_path):
        report = lib.verify_chain(tmp_path / ".hermes" / "receipts" / "runs.ndjson")
        assert report["records"] == 0
        assert report["errors"] == ["no receipts file found"]


class TestCommands:
    def test_slash_latest(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs)
        out = stubs["commands"]["receipts"]("latest")
        assert '"outcome": "completed"' in out

    def test_slash_stats(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs, "a")
        _run_one_turn(stubs, "b")
        out = stubs["commands"]["receipts"]("stats")
        assert "Runs: 2" in out
        assert "completed: 2" in out

    def test_slash_verify(self, plugin, tmp_path):
        _mod, stubs = plugin
        _run_one_turn(stubs)
        out = stubs["commands"]["receipts"]("verify")
        assert "OK" in out and "chain intact" in out

    def test_slash_empty(self, plugin):
        _mod, stubs = plugin
        out = stubs["commands"]["receipts"]("latest")
        assert out == "No receipts recorded yet."

    def test_cli_handler_verify_exit_codes(self, plugin, tmp_path):
        _mod, stubs = plugin
        _setup, handler = stubs["cli"]["receipts"]
        _run_one_turn(stubs)

        class _Args:
            action = "verify"
            file = None

        assert handler(_Args()) == 0
        path = _receipts_file(tmp_path)
        rec = json.loads(path.read_text().strip())
        rec["model"] = "forged"
        path.write_text(json.dumps(rec, sort_keys=True) + "\n")
        assert handler(_Args()) == 1

    def test_concurrent_processes_do_not_fork_chain(self, lib, tmp_path):
        """Two writers appending receipts under the lock produce a valid chain."""
        import multiprocessing

        path_dir = tmp_path / ".hermes"

        def _write_n(n):
            for i in range(n):
                lib.write_receipt(
                    path_dir, {"v": 1, "run_id": f"{n}-{i}", "outcome": "completed"}
                )

        procs = [multiprocessing.Process(target=_write_n, args=(5,)) for _ in range(3)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0
        report = lib.verify_chain(path_dir / "receipts" / "runs.ndjson")
        assert report["ok"] is True
        assert report["records"] == 15
