"""Tests for the final two eksikler (v1.0 closure):

* B1 — `evaluate_batch` actually routes through `distributed.WorkerBackend`
  when one is supplied, and the `--workers raysim` CLI path reaches
  that dispatch.
* C4 — VS Code extension package.json is structurally valid JSON with
  the expected command ids and configuration keys, and `extension.ts`
  registers each command.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import algorithms    # noqa: E402
import distributed   # noqa: E402
import evaluator     # noqa: E402
import evolver       # noqa: E402


# ---------------------------------------------------------------------------
# B1 — evaluate_batch backend dispatch
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def _pop(self, scores):
        return [
            algorithms.Individual(cid=f"c{i}", genome=str(s), fitness=float("nan"))
            for i, s in enumerate(scores)
        ]

    def test_local_path_still_uses_semaphore(self):
        @evaluator.fitness_spec(timeout_s=2)
        def fit(c, ctx): return float(len(c))
        pop = self._pop([1, 22, 333])
        asyncio.run(evaluator.evaluate_batch(pop, fit, concurrency=2))
        assert [ind.fitness for ind in pop] == [1.0, 2.0, 3.0]

    def test_raysim_backend_is_used_when_supplied(self):
        """Backend.map is called N times, once per candidate."""
        calls: list[str] = []

        class RecordingBackend:
            name = "record"
            async def map(self, fn, items):
                calls.extend(i.genome for i in items)
                out = []
                for item in items:
                    res = fn(item)
                    if asyncio.iscoroutine(res):
                        res = await res
                    out.append(res)
                return out

        @evaluator.fitness_spec(timeout_s=2)
        def fit(c, ctx): return float(len(c))
        pop = self._pop(["xx", "yyy"])
        asyncio.run(evaluator.evaluate_batch(
            pop, fit, concurrency=1, backend=RecordingBackend(),
        ))
        assert calls == ["xx", "yyy"]
        assert [ind.fitness for ind in pop] == [2.0, 3.0]

    def test_select_raysim_returns_backend(self):
        backend = distributed.select_backend("raysim", workers=2)
        assert isinstance(backend, distributed.RaySimBackend)


# ---------------------------------------------------------------------------
# CLI flag --workers actually routes
# ---------------------------------------------------------------------------


class TestWorkersFlagRouting:
    def test_workers_raysim_end_to_end(self, tmp_path, monkeypatch, capsys):
        """`evolver run --workers raysim` completes and assigns fitness."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        exp = tmp_path / "exp"
        (exp / "seed").mkdir(parents=True)
        (exp / "logs").mkdir()
        (exp / "seed" / "initial.txt").write_text("seed prompt\n")
        (exp / "fitness.py").write_text(
            "def fitness(c, ctx): return float(len(c)) / 20.0\n"
        )
        (exp / "evolver_sdk.py").write_text(
            "from evaluator import fitness_spec\n"
        )
        import storage as _storage
        _storage.open_db(exp / "lineage.db").close()

        # Mock httpx so operator LLM calls return a stable string.
        import httpx
        original = httpx.AsyncClient.post
        async def fake(self, url, json=None, **kw):
            class _R:
                status_code = 200
                headers: dict = {}
                def raise_for_status(self_): pass
                def json(self_inner):
                    return {
                        "choices": [{"message": {"content": "evolved prompt text"}}],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                    }
            return _R()
        httpx.AsyncClient.post = fake  # type: ignore[assignment]
        try:
            import argparse
            args = argparse.Namespace(
                dir=str(exp),
                generations=1, pop=2, budget=0.0,
                algorithm="es", concurrency=1, seed=0,
                input_rate=0.0, output_rate=0.0, no_cache=True,
                descriptor_controller="off", descriptor_every_k=5,
                bandit_director="off",
                workers="raysim",
            )
            result = asyncio.run(evolver._run_loop(args, exp))
        finally:
            httpx.AsyncClient.post = original  # type: ignore[assignment]

        assert result["ok"]
        conn = _storage.open_db(exp / "lineage.db")
        rows = _storage.get_best(conn, "fitness", k=1)
        conn.close()
        assert rows and rows[0]["value"] > 0


# ---------------------------------------------------------------------------
# C4 — VS Code extension package structural validation
# ---------------------------------------------------------------------------


EXTENSION_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "editor" / "vscode"
)


class TestVsCodeExtension:
    def test_package_json_parses_and_declares_commands(self):
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text("utf-8"))
        assert pkg["name"] == "darwinian-evolver"
        cmd_ids = {c["command"] for c in pkg["contributes"]["commands"]}
        expected = {
            "darwinianEvolver.run",
            "darwinianEvolver.dashboard",
            "darwinianEvolver.synthesiseFitness",
            "darwinianEvolver.showLineage",
            "darwinianEvolver.acceptHumanEdit",
        }
        assert expected.issubset(cmd_ids)

    def test_activation_events_match_commands(self):
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text("utf-8"))
        events = set(pkg["activationEvents"])
        for cmd in pkg["contributes"]["commands"]:
            assert f"onCommand:{cmd['command']}" in events

    def test_configuration_settings_exposed(self):
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text("utf-8"))
        props = pkg["contributes"]["configuration"]["properties"]
        for key in ("darwinianEvolver.cliPath",
                    "darwinianEvolver.pythonPath",
                    "darwinianEvolver.dashboardHost",
                    "darwinianEvolver.dashboardPort"):
            assert key in props

    def test_extension_ts_registers_every_command(self):
        src = (EXTENSION_DIR / "src" / "extension.ts").read_text("utf-8")
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text("utf-8"))
        for cmd in pkg["contributes"]["commands"]:
            assert cmd["command"] in src, f"command {cmd['command']} not registered in extension.ts"

    def test_tsconfig_is_valid_json(self):
        # VS Code uses tsconfig.json — must be strict JSON (no comments)
        # because we feed it directly to tsc.
        tsc = json.loads((EXTENSION_DIR / "tsconfig.json").read_text("utf-8"))
        assert tsc["compilerOptions"]["outDir"] == "out"
