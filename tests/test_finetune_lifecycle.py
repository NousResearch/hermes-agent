"""
Lifecycle tests for the finetune pipeline review fixes.

Covers: eval gate metric intersection (incl. empty-intersection fail),
exp-based perplexity, baseline lookup, config generation guards, route
artifact selection + state cache invalidation, bench-gate stale-result
rejection, the accepted-baseline pointer, the serving manifest +
scale-based plugin routing, first-adapter deactivation, dead-PID health
checks, atomic GGUF conversion, pipeline locking, cron wiring, gc
retention policy, and llama-server stop/start safety. Everything runs
against tmp dirs and fake executables — no real
accelerate/docker/llama-server.
"""

import json
import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

# Add skill scripts to path
_scripts_dir = str(Path(__file__).resolve().parent.parent / "optional-skills" / "mlops" / "finetune" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ft_home(tmp_path, monkeypatch):
    """Point every finetune path constant at a temp HERMES_HOME.

    The skill modules do `from common import X` at import time, so the
    constants must be patched on each importing module, not just common.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import common
    import eval as eval_mod
    import manage
    import route
    import train

    ft = home / "finetune"
    paths = {
        "HERMES_HOME": home,
        "FINETUNE_DIR": ft,
        "DATA_DIR": ft / "data",
        "EXTRACTED_DIR": ft / "data" / "extracted",
        "SCORED_DIR": ft / "data" / "scored",
        "CLUSTERS_DIR": ft / "data" / "clusters",
        "IMPORTED_DIR": ft / "data" / "imported",
        "ADAPTERS_DIR": ft / "adapters",
        "MODELS_DIR": ft / "models" / "merged",
        "LOGS_DIR": ft / "logs",
        "BENCH_DIR": ft / "bench",
        "FEEDBACK_PATH": ft / "feedback.jsonl",
        "REGISTRY_PATH": ft / "adapters" / "registry.json",
        "CLUSTER_STATE_PATH": ft / "adapters" / "cluster_state.json",
        "EXTRACT_STATE_PATH": ft / "extract_state.json",
        "LOCK_PATH": ft / "finetune.lock",
    }
    for mod in (common, eval_mod, manage, route, train):
        for name, value in paths.items():
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, value)

    common.ensure_dirs()
    return home


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ============================================================================
# eval.py — verdict metric intersection
# ============================================================================

class TestVerdictIntersection:
    def test_missing_gate_metrics_are_skipped_not_failed(self, capsys):
        """trainer_state-only metrics (eval_loss/perplexity) must not fail
        format_compliance, nor auto-pass hallucination_rate."""
        from eval import compare_metrics, verdict

        baseline = {"eval_loss": 1.00, "perplexity": math.exp(1.00)}
        candidate = {"eval_loss": 0.95, "perplexity": math.exp(0.95)}

        checks = verdict(compare_metrics(candidate, baseline))

        # Only the core loss/perplexity gates apply.
        assert checks["overall"] is True
        assert checks["eval_loss"] is True
        assert checks["perplexity"] is True
        assert "format_compliance" not in checks
        assert "no_hallucinations" not in checks
        assert "tool_selection" not in checks

        out = capsys.readouterr().out
        assert "format_compliance" in out
        assert "skipped" in out

    def test_metric_missing_from_baseline_is_skipped(self):
        """hallucination_rate present only in the candidate must not gate
        (it used to fail OPEN when the baseline lacked the key)."""
        from eval import compare_metrics, verdict

        baseline = {"tool_selection_accuracy": 0.8}
        candidate = {"tool_selection_accuracy": 0.8, "hallucination_rate": 0.5}

        checks = verdict(compare_metrics(candidate, baseline))
        assert "no_hallucinations" not in checks
        assert checks["tool_selection"] is True
        assert checks["overall"] is True

    def test_present_metrics_still_gate(self):
        from eval import compare_metrics, verdict

        baseline = {"hallucination_rate": 0.0, "eval_loss": 1.0}
        candidate = {"hallucination_rate": 0.10, "eval_loss": 1.0}

        checks = verdict(compare_metrics(candidate, baseline))
        assert checks["no_hallucinations"] is False
        assert checks["overall"] is False

    def test_boundary_exactly_at_threshold_passes(self):
        from eval import THRESHOLDS, verdict

        # Delta-based gates get hand-built comparison entries so the
        # boundary is exact (0.77 - 0.80 in floats lands just below -0.03).
        comparison = {
            "tool_selection_accuracy": {
                "baseline": 0.80, "candidate": 0.77,
                "delta": -THRESHOLDS["tool_selection_accuracy"],
            },
            "eval_loss": {
                "baseline": 1.00, "candidate": 1.05,
                "delta": THRESHOLDS["eval_loss_regression"],
            },
            # Value-based gates: exactly the minimum compliance / exactly
            # the allowed relative perplexity regression.
            "format_compliance": {
                "baseline": 0.99,
                "candidate": THRESHOLDS["format_compliance_min"],
                "delta": THRESHOLDS["format_compliance_min"] - 0.99,
            },
            "perplexity": {
                "baseline": 10.0,
                "candidate": 10.0 * (1 + THRESHOLDS["perplexity_regression"]),
                "delta": 10.0 * THRESHOLDS["perplexity_regression"],
            },
        }

        checks = verdict(comparison)
        assert checks["tool_selection"] is True
        assert checks["format_compliance"] is True
        assert checks["eval_loss"] is True
        assert checks["perplexity"] is True
        assert checks["overall"] is True

    def test_just_past_threshold_fails(self):
        from eval import compare_metrics, verdict

        baseline = {"eval_loss": 1.00}
        candidate = {"eval_loss": 1.06}  # > 0.05 allowed regression

        checks = verdict(compare_metrics(candidate, baseline))
        assert checks["eval_loss"] is False
        assert checks["overall"] is False


# ============================================================================
# eval.py — perplexity, baseline lookup, loud gate skip
# ============================================================================

class TestEvalMetrics:
    def test_perplexity_uses_natural_exp(self, ft_home):
        import common
        from eval import EvalGate

        adapter_dir = common.ADAPTERS_DIR / "c1" / "v1"
        _write_json(adapter_dir / "adapter_model" / "trainer_state.json",
                    {"best_metric": 1.0})

        metrics = EvalGate()._extract_training_metrics(adapter_dir)
        assert metrics["eval_loss"] == 1.0
        assert metrics["perplexity"] == pytest.approx(math.e)

    def test_find_baseline_exact_cluster_and_mtime(self, ft_home):
        import common
        import eval as eval_mod

        results = common.BENCH_DIR / "results"

        # A different cluster that shares the prefix — must NOT match "c1".
        _write_json(results / "c1_extra_v1_20240101_000000.json",
                    {"metrics": {"eval_loss": 9.9}})
        assert eval_mod._find_baseline("c1") is None

        older = results / "c1_v1_20240101_000000.json"
        newer = results / "c1_v2_20240202_000000.json"
        _write_json(older, {"metrics": {"eval_loss": 1.0}})
        _write_json(newer, {"metrics": {"eval_loss": 2.0}})
        now = time.time()
        os.utime(older, (now - 100, now - 100))
        os.utime(newer, (now, now))

        found = eval_mod._find_baseline("c1")
        assert found["metrics"]["eval_loss"] == 2.0

        # A newer gate-skipped result (empty metrics) must not shadow the
        # real baseline.
        skipped = results / "c1_v3_20240303_000000.json"
        _write_json(skipped, {"metrics": {}, "gate_skipped": True})
        os.utime(skipped, (now + 100, now + 100))
        found = eval_mod._find_baseline("c1")
        assert found["metrics"]["eval_loss"] == 2.0

    def test_no_metrics_gate_skip_is_loud_and_recorded(self, ft_home, capsys):
        import common
        from eval import EvalGate

        (common.ADAPTERS_DIR / "c1" / "v1").mkdir(parents=True)
        passed, report = EvalGate().evaluate("c1", "v1")

        assert passed is True
        assert "gate skipped" in report
        out = capsys.readouterr().out
        assert "no eval metrics found — gate skipped, treating as pass" in out

        saved = list((common.BENCH_DIR / "results").glob("c1_v1_*.json"))
        assert len(saved) == 1
        data = json.loads(saved[0].read_text())
        assert data["gate_skipped"] is True
        assert "gate skipped" in data["note"]


# ============================================================================
# train.py — generate_config guards
# ============================================================================

class TestGenerateConfig:
    def _orchestrator(self, base_model="kai-os/Carnice-9b"):
        from train import TrainingOrchestrator
        return TrainingOrchestrator(config={"base_model": base_model,
                                            "chat_template": "chatml"})

    def _write_train_data(self, cluster_id, n):
        import train as train_mod
        path = train_mod.CLUSTERS_DIR / cluster_id / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"conversations": [
                    {"from": "human", "value": f"q{i}"},
                    {"from": "gpt", "value": f"a{i}"},
                ]}) + "\n")
        return path

    def test_gguf_base_model_rejected(self, ft_home):
        self._write_train_data("c1", 10)
        orch = self._orchestrator(base_model="~/programs/carnice/Carnice-9b-Q8_0.gguf")
        with pytest.raises(ValueError, match="GGUF"):
            orch.generate_config("c1", "v1")

    def test_tiny_dataset_disables_eval_split(self, ft_home):
        import train as train_mod

        self._write_train_data("c1", 10)
        config_path = self._orchestrator().generate_config("c1", "v1")
        cfg = yaml.safe_load(config_path.read_text())

        assert cfg["val_set_size"] == 0
        for key in ("eval_steps", "early_stopping_patience",
                    "metric_for_best_model", "greater_is_better"):
            assert key not in cfg
        # sanity: base model made it through untouched
        assert cfg["base_model"] == "kai-os/Carnice-9b"

        manifest = json.loads(
            (train_mod.ADAPTERS_DIR / "c1" / "v1" / "dataset_manifest.json").read_text()
        )
        assert manifest["train_size"] == 10

    def test_large_dataset_keeps_eval_split(self, ft_home):
        self._write_train_data("c2", 50)  # exactly the threshold → split kept
        config_path = self._orchestrator().generate_config("c2", "v1")
        cfg = yaml.safe_load(config_path.read_text())

        assert cfg["val_set_size"] == 0.1
        assert cfg["eval_steps"] == 50
        assert cfg["metric_for_best_model"] == "eval_loss"

    def test_merge_and_quantize_removed(self):
        from train import TrainingOrchestrator
        assert not hasattr(TrainingOrchestrator, "merge_and_quantize")


# ============================================================================
# route.py — artifact selection
# ============================================================================

class TestRouteArtifactSelection:
    def _setup_cluster(self, cluster_id="c1", version="v1"):
        import route as route_mod
        _write_json(route_mod.CLUSTER_STATE_PATH, {
            "centroids": {cluster_id: [1.0, 0.0]},
            "clusters": {cluster_id: {"label": "dev"}},
        })
        _write_json(route_mod.REGISTRY_PATH, {"adapters": [
            {"cluster_id": cluster_id, "version": version, "status": "active"},
        ]})
        adapter_dir = route_mod.ADAPTERS_DIR / cluster_id / version
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return adapter_dir

    def _router(self, monkeypatch):
        import numpy as np
        from route import AdapterRouter

        router = AdapterRouter(config={
            "clustering": {"confidence_threshold": 0.5},
            "routing": {"enabled": True},
        })

        class FakeModel:
            def encode(self, prompts, normalize_embeddings=True):
                return np.array([[1.0, 0.0]])

        monkeypatch.setattr(router, "_get_embed_model", lambda: FakeModel())
        return router

    def test_prefers_adapter_gguf(self, ft_home, monkeypatch):
        adapter_dir = self._setup_cluster()
        (adapter_dir / "adapter_model").mkdir()
        (adapter_dir / "adapter.gguf").write_text("gguf")

        result = self._router(monkeypatch).route("write some code")
        assert result["cluster_id"] == "c1"
        assert result["adapter_path"] == str(adapter_dir / "adapter.gguf")
        assert result["fallback"] is False

    def test_peft_only_returns_none_with_hint(self, ft_home, monkeypatch, caplog):
        adapter_dir = self._setup_cluster()
        (adapter_dir / "adapter_model").mkdir()  # PEFT safetensors only

        with caplog.at_level("WARNING", logger="hermes.finetune"):
            result = self._router(monkeypatch).route("write some code")

        assert result["cluster_id"] == "c1"
        assert result["adapter_path"] is None
        assert any("redeploy" in rec.message for rec in caplog.records)

    def test_resolver_never_returns_safetensors_dir(self, tmp_path):
        from route import _resolve_adapter_artifact

        adapter_dir = tmp_path / "v1"
        (adapter_dir / "adapter_model").mkdir(parents=True)
        assert _resolve_adapter_artifact(adapter_dir) is None

        (adapter_dir / "adapter.gguf").write_text("gguf")
        assert _resolve_adapter_artifact(adapter_dir) == str(adapter_dir / "adapter.gguf")


# ============================================================================
# manage.py — bench gate
# ============================================================================

class TestRunBench:
    def _setup_bench(self, ft_home, monkeypatch, script_body: str):
        import common
        import manage

        results = common.BENCH_DIR / "results"
        results.mkdir(parents=True, exist_ok=True)

        env_script = ft_home / "fake_bench_env.py"
        env_script.write_text(script_body)
        config = ft_home / "bench_default.yaml"
        config.write_text("cases: []\n")

        monkeypatch.setattr(manage, "BENCH_ENV_SCRIPT", env_script)
        monkeypatch.setattr(manage, "BENCH_DEFAULT_CONFIG", config)
        monkeypatch.setattr(manage, "BENCH_ASSETS_DIR", ft_home)
        return results

    def test_nonzero_exit_never_reuses_stale_result(self, ft_home, monkeypatch):
        import manage

        results = self._setup_bench(
            ft_home, monkeypatch,
            "import sys; print('bench exploded: traceback...'); sys.exit(3)\n",
        )
        stale = results / "bench_20240101_000000.json"
        _write_json(stale, {"metrics": {"total_cases": 100}})

        assert manage.run_bench() is None
        assert stale.exists()  # untouched, just not reused

    def test_zero_exit_without_new_result_fails(self, ft_home, monkeypatch):
        import manage

        results = self._setup_bench(ft_home, monkeypatch, "import sys; sys.exit(0)\n")
        _write_json(results / "bench_20240101_000000.json",
                    {"metrics": {"total_cases": 100}})

        assert manage.run_bench() is None

    def test_new_result_is_returned(self, ft_home, monkeypatch):
        import manage

        script = (
            "import json, pathlib, sys\n"
            f"out = pathlib.Path({str(ft_home)!r}) / 'finetune' / 'bench' / 'results' / 'bench_new.json'\n"
            "out.write_text(json.dumps({'metrics': {'total_cases': 100}}))\n"
            "sys.exit(0)\n"
        )
        results = self._setup_bench(ft_home, monkeypatch, script)
        old = results / "bench_20240101_000000.json"
        _write_json(old, {"metrics": {"total_cases": 100}})

        result = manage.run_bench()
        assert result is not None
        assert result.name == "bench_new.json"


# ============================================================================
# manage.py — gc retention policy
# ============================================================================

class TestGcRetention:
    def _setup_versions(self, ft_home):
        import common

        cluster_dir = common.ADAPTERS_DIR / "c1"
        for v in ("v1", "v2", "v3", "v4", "v5"):
            (cluster_dir / v).mkdir(parents=True)
            (cluster_dir / v / "adapter.gguf").write_text("x")

        _write_json(common.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v1", "status": "trained"},
            {"cluster_id": "c1", "version": "v2", "status": "trained"},
            {"cluster_id": "c1", "version": "v3", "status": "previous"},
            {"cluster_id": "c1", "version": "v4", "status": "active",
             "rollback_target": "v3"},
            {"cluster_id": "c1", "version": "v5", "status": "trained"},
        ]})
        return cluster_dir

    def test_gc_actually_deletes_despite_full_registry(self, ft_home):
        """Every version is registered (the pipeline registers everything),
        yet gc must still delete beyond active + rollback + N recent."""
        from manage import AdapterRegistry

        cluster_dir = self._setup_versions(ft_home)
        registry = AdapterRegistry()
        removed = registry.gc(keep_versions=1)

        # Protected: v4 (active) + v3 (rollback target). Recent: v5.
        assert sorted(removed) == ["c1/v1", "c1/v2"]
        assert not (cluster_dir / "v1").exists()
        assert not (cluster_dir / "v2").exists()
        for kept in ("v3", "v4", "v5"):
            assert (cluster_dir / kept).exists()

        # Deleted versions dropped from the registry too.
        remaining = {a["version"] for a in registry.registry["adapters"]}
        assert remaining == {"v3", "v4", "v5"}

    def test_gc_keep_two(self, ft_home):
        from manage import AdapterRegistry

        cluster_dir = self._setup_versions(ft_home)
        removed = AdapterRegistry().gc(keep_versions=2)

        assert removed == ["c1/v1"]
        for kept in ("v2", "v3", "v4", "v5"):
            assert (cluster_dir / kept).exists()


# ============================================================================
# manage.py — llama-server lifecycle safety
# ============================================================================

class TestLlamaServerLifecycle:
    def test_stop_refuses_mismatched_cmdline(self, tmp_path):
        """A reused PID belonging to an unrelated process must not be killed."""
        from manage import stop_llama_server

        proc = subprocess.Popen(["sleep", "60"])
        try:
            pid_file = tmp_path / "server.pid"
            pid_file.write_text(str(proc.pid))

            result = stop_llama_server(pid_file, expected_basename="llama-server")

            assert result is False
            assert proc.poll() is None  # unrelated process left alive
            assert not pid_file.exists()  # stale pid file removed
        finally:
            proc.kill()
            proc.wait()

    def test_stop_refuses_lookalike_basename(self, tmp_path):
        """A recycled PID running `tail -f .../llama-server.log` used to
        pass the substring identity check and get killed."""
        from manage import stop_llama_server

        log = tmp_path / "llama-server.log"
        log.write_text("")
        proc = subprocess.Popen(["tail", "-f", str(log)])
        try:
            pid_file = tmp_path / "server.pid"
            pid_file.write_text(str(proc.pid))

            result = stop_llama_server(pid_file, expected_basename="llama-server")

            assert result is False
            assert proc.poll() is None  # tail survives
            assert not pid_file.exists()
        finally:
            proc.kill()
            proc.wait()

    def test_stop_kills_matching_process(self, tmp_path):
        from manage import stop_llama_server

        # Python wrapper so /proc/<pid>/cmdline reliably contains the
        # script path (a /bin/sh script may exec-optimize into `sleep`).
        exe = tmp_path / "llama-server"
        exe.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)

        proc = subprocess.Popen([str(exe)])
        try:
            # Right after fork, /proc/<pid>/cmdline can be empty until the
            # execve completes — wait for the identity to become visible.
            deadline = time.time() + 5
            while time.time() < deadline:
                if b"llama-server" in Path(f"/proc/{proc.pid}/cmdline").read_bytes():
                    break
                time.sleep(0.05)

            pid_file = tmp_path / "server.pid"
            pid_file.write_text(str(proc.pid))

            result = stop_llama_server(pid_file, expected_basename="llama-server")

            assert result is True
            proc.wait(timeout=5)
            assert proc.poll() is not None
            assert not pid_file.exists()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def _redeploy_cfg(self, ft_home, server_command):
        snapshot = ft_home / "snap"
        snapshot.mkdir(exist_ok=True)
        return {
            "serving": {
                "auto_redeploy": True,
                "converter": str(ft_home / "convert_lora_to_gguf.py"),
                "base_model_snapshot": str(snapshot),
                "server_command": server_command,
                "server_pid_file": str(ft_home / "llama.pid"),
                "server_log_path": str(ft_home / "llama.log"),
                "health_check_url": "http://localhost:1/health",
                "health_check_timeout": 1,
            },
            "training": {"base_model": "kai-os/Carnice-9b"},
        }

    def _adapter_with_cached_gguf(self, ft_home):
        import common
        adapter_dir = common.ADAPTERS_DIR / "c1" / "v1"
        (adapter_dir / "adapter_model").mkdir(parents=True)
        (adapter_dir / "adapter.gguf").write_text("gguf")
        return adapter_dir

    def test_redeploy_validates_exe_before_stopping(self, ft_home, monkeypatch):
        """A missing server binary must abort BEFORE the old server is
        stopped — never leave the user with no server at all."""
        import manage

        adapter_dir = self._adapter_with_cached_gguf(ft_home)
        cfg = self._redeploy_cfg(
            ft_home, str(ft_home / "does-not-exist" / "llama-server") + " -m base.gguf",
        )
        monkeypatch.setattr(manage, "load_config", lambda: cfg)

        calls = []
        monkeypatch.setattr(manage, "stop_llama_server",
                            lambda *a, **k: calls.append("stop"))
        monkeypatch.setattr(manage, "start_llama_server",
                            lambda *a, **k: (calls.append("start"), 4242)[1])

        assert manage.redeploy(adapter_dir) is False
        assert calls == []  # neither stop nor start happened

    def test_redeploy_stop_then_start_when_exe_valid(self, ft_home, monkeypatch):
        import manage

        adapter_dir = self._adapter_with_cached_gguf(ft_home)
        exe = ft_home / "llama-server"
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        cfg = self._redeploy_cfg(ft_home, f"{exe} -m base.gguf")
        monkeypatch.setattr(manage, "load_config", lambda: cfg)

        calls = []
        monkeypatch.setattr(manage, "stop_llama_server",
                            lambda *a, **k: calls.append("stop"))
        monkeypatch.setattr(manage, "start_llama_server",
                            lambda *a, **k: (calls.append("start"), 4242)[1])
        monkeypatch.setattr(manage, "health_check_llama_server",
                            lambda *a, **k: True)

        assert manage.redeploy(adapter_dir) is True
        assert calls == ["stop", "start"]

    def test_redeploy_requires_converter_config(self, ft_home, monkeypatch):
        import manage

        adapter_dir = self._adapter_with_cached_gguf(ft_home)
        # Remove the cached GGUF so conversion would actually be needed.
        (adapter_dir / "adapter.gguf").unlink()
        cfg = self._redeploy_cfg(ft_home, "")
        cfg["serving"]["converter"] = ""
        monkeypatch.setattr(manage, "load_config", lambda: cfg)

        assert manage.redeploy(adapter_dir) is False

    def test_build_server_cmd_expands_tilde(self):
        from manage import build_server_cmd

        cmd = build_server_cmd(
            "~/bin/llama-server -m ~/models/base.gguf", Path("/x/adapter.gguf"),
        )
        assert cmd[0] == str(Path("~/bin/llama-server").expanduser())
        assert cmd[2] == str(Path("~/models/base.gguf").expanduser())
        assert "--lora" in cmd and "/x/adapter.gguf" in cmd

    def test_build_server_cmd_quotes_spaces(self):
        """%LORA% substitution must keep a path with spaces as ONE argv
        token (the old code shlex.split a raw substitution)."""
        from manage import build_server_cmd

        cmd = build_server_cmd(
            "llama-server -m base.gguf --lora %LORA% --port 8008",
            Path("/tmp/my adapters/a.gguf"),
        )
        i = cmd.index("--lora")
        assert cmd[i + 1] == "/tmp/my adapters/a.gguf"
        assert cmd[-2:] == ["--port", "8008"]

    def test_build_server_cmd_base_model_strips_lora(self):
        """lora_path=None derives the base-model command by stripping the
        --lora %LORA% segment (used after deactivating a first adapter)."""
        from manage import build_server_cmd

        cmd = build_server_cmd(
            "llama-server -m base.gguf --lora %LORA% --port 8008", None,
        )
        assert "--lora" not in cmd and "%LORA%" not in cmd
        assert cmd == ["llama-server", "-m", "base.gguf", "--port", "8008"]

        # Template without %LORA% is used as-is.
        cmd = build_server_cmd("llama-server -m base.gguf", None)
        assert cmd == ["llama-server", "-m", "base.gguf"]


# ============================================================================
# manage.py — serving manifest (written on verified deploy, cleared on stop)
# ============================================================================

class TestServingManifest:
    def _cfg(self, ft_home, exe):
        snapshot = ft_home / "snap"
        snapshot.mkdir(exist_ok=True)
        return {
            "serving": {
                "auto_redeploy": True,
                "converter": str(ft_home / "convert_lora_to_gguf.py"),
                "base_model_snapshot": str(snapshot),
                "server_command": f"{exe} -m base.gguf --lora %LORA%",
                "server_pid_file": str(ft_home / "llama.pid"),
                "server_log_path": str(ft_home / "llama.log"),
                "health_check_url": "http://localhost:8008/v1/models",
                "health_check_timeout": 1,
            },
            "training": {"base_model": "kai-os/Carnice-9b"},
        }

    def _exe(self, ft_home):
        exe = ft_home / "llama-server"
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        return exe

    def _adapter(self, ft_home, cluster="c1", version="v1"):
        import common
        adapter_dir = common.ADAPTERS_DIR / cluster / version
        (adapter_dir / "adapter_model").mkdir(parents=True)
        (adapter_dir / "adapter.gguf").write_text("gguf")
        return adapter_dir

    def test_successful_redeploy_writes_manifest(self, ft_home, monkeypatch):
        import manage

        adapter_dir = self._adapter(ft_home)
        monkeypatch.setattr(manage, "load_config",
                            lambda: self._cfg(ft_home, self._exe(ft_home)))
        monkeypatch.setattr(manage, "stop_llama_server", lambda *a, **k: True)
        monkeypatch.setattr(manage, "start_llama_server", lambda *a, **k: 4242)
        monkeypatch.setattr(manage, "health_check_llama_server",
                            lambda *a, **k: True)

        assert manage.redeploy(adapter_dir) is True

        manifest_path = manage.serving_manifest_path()
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert set(data) == {"updated_at", "server", "adapters"}
        assert data["server"] == {
            "pid": 4242, "health_url": "http://localhost:8008/v1/models",
        }
        assert data["adapters"] == [{
            "id": 0, "cluster": "c1", "version": "v1",
            "gguf": str(adapter_dir / "adapter.gguf"),
        }]

    def test_failed_health_check_clears_manifest(self, ft_home, monkeypatch):
        """After the old server is stopped, a failed redeploy must never
        leave a manifest claiming an adapter is being served."""
        import manage

        adapter_dir = self._adapter(ft_home)
        # A manifest from a previous successful deploy.
        manage.write_serving_manifest(
            1111, "http://localhost:8008/v1/models",
            [{"id": 0, "cluster": "c1", "version": "v0", "gguf": "/old.gguf"}],
        )
        monkeypatch.setattr(manage, "load_config",
                            lambda: self._cfg(ft_home, self._exe(ft_home)))
        monkeypatch.setattr(manage, "stop_llama_server", lambda *a, **k: True)
        monkeypatch.setattr(manage, "start_llama_server", lambda *a, **k: 4242)
        monkeypatch.setattr(manage, "health_check_llama_server",
                            lambda *a, **k: False)

        assert manage.redeploy(adapter_dir) is False
        assert not manage.serving_manifest_path().exists()

    def test_exe_validation_failure_keeps_manifest(self, ft_home, monkeypatch):
        """A redeploy aborted BEFORE stopping the old server leaves the old
        server (and its manifest) untouched."""
        import manage

        adapter_dir = self._adapter(ft_home)
        manage.write_serving_manifest(
            1111, "http://localhost:8008/v1/models",
            [{"id": 0, "cluster": "c1", "version": "v0", "gguf": "/old.gguf"}],
        )
        cfg = self._cfg(ft_home, str(ft_home / "missing" / "llama-server"))
        monkeypatch.setattr(manage, "load_config", lambda: cfg)

        assert manage.redeploy(adapter_dir) is False
        assert manage.serving_manifest_path().exists()

    def test_redeploy_base_writes_adapterless_manifest(self, ft_home, monkeypatch):
        import manage

        monkeypatch.setattr(manage, "load_config",
                            lambda: self._cfg(ft_home, self._exe(ft_home)))
        monkeypatch.setattr(manage, "stop_llama_server", lambda *a, **k: True)
        started = []
        monkeypatch.setattr(
            manage, "start_llama_server",
            lambda template, lora, *a, **k: (started.append(lora), 4242)[1],
        )
        monkeypatch.setattr(manage, "health_check_llama_server",
                            lambda *a, **k: True)

        assert manage.redeploy_base() is True
        assert started == [None]  # no LoRA handed to the server
        data = json.loads(manage.serving_manifest_path().read_text())
        assert data["adapters"] == []


# ============================================================================
# manage.py — health check must verify the LAUNCHED process survived
# ============================================================================

class TestHealthCheckLaunchedPid:
    def _dead_pid(self):
        proc = subprocess.Popen(["sleep", "0"])
        proc.wait()
        return proc.pid

    def _fake_ok_response(self, monkeypatch):
        import urllib.request

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())

    def test_dead_pid_fails_even_when_url_responds(self, monkeypatch):
        """An old server on the same port answers the health URL while the
        launched process died on 'address in use' — that is a FAILURE."""
        from manage import health_check_llama_server

        self._fake_ok_response(monkeypatch)
        start = time.time()
        result = health_check_llama_server(
            "http://localhost:8008/v1/models", timeout=10, pid=self._dead_pid(),
        )
        assert result is False
        assert time.time() - start < 5  # fails fast, not after the timeout

    def test_live_pid_and_responding_url_passes(self, monkeypatch):
        from manage import health_check_llama_server

        self._fake_ok_response(monkeypatch)
        assert health_check_llama_server(
            "http://localhost:8008/v1/models", timeout=5, pid=os.getpid(),
        ) is True


# ============================================================================
# manage.py — GGUF conversion is atomic (temp file + os.replace)
# ============================================================================

class TestConvertAtomicity:
    def _setup(self, tmp_path):
        adapter_dir = tmp_path / "v1"
        (adapter_dir / "adapter_model").mkdir(parents=True)
        snapshot = tmp_path / "snap"
        snapshot.mkdir()
        return adapter_dir, snapshot

    def _converter(self, tmp_path, body):
        converter = tmp_path / "converter.py"
        converter.write_text(body)
        return converter

    def test_failed_conversion_never_leaves_gguf(self, tmp_path):
        """A converter that dies mid-write must not leave a truncated
        adapter.gguf that the exists() cache check would trust later."""
        from manage import convert_adapter_to_gguf

        adapter_dir, snapshot = self._setup(tmp_path)
        converter = self._converter(tmp_path, (
            "import sys\n"
            "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
            "open(out, 'w').write('truncated garbage')\n"
            "sys.exit(1)\n"
        ))

        with pytest.raises(RuntimeError, match="conversion failed"):
            convert_adapter_to_gguf(adapter_dir, snapshot, converter)
        assert not (adapter_dir / "adapter.gguf").exists()
        assert not (adapter_dir / "adapter.gguf.converting").exists()

    def test_timeout_is_clean_failure_without_gguf(self, tmp_path):
        """A hung converter becomes a RuntimeError (not a raw
        TimeoutExpired traceback) and leaves nothing behind."""
        from manage import convert_adapter_to_gguf

        adapter_dir, snapshot = self._setup(tmp_path)
        converter = self._converter(tmp_path, (
            "import sys, time\n"
            "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
            "open(out, 'w').write('partial')\n"
            "time.sleep(30)\n"
        ))

        with pytest.raises(RuntimeError, match="timed out"):
            convert_adapter_to_gguf(adapter_dir, snapshot, converter, timeout=1)
        assert not (adapter_dir / "adapter.gguf").exists()
        assert not (adapter_dir / "adapter.gguf.converting").exists()

    def test_success_moves_output_into_place(self, tmp_path):
        from manage import convert_adapter_to_gguf

        adapter_dir, snapshot = self._setup(tmp_path)
        converter = self._converter(tmp_path, (
            "import sys\n"
            "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
            "open(out, 'w').write('gguf-bytes')\n"
        ))

        result = convert_adapter_to_gguf(adapter_dir, snapshot, converter)
        assert result == adapter_dir / "adapter.gguf"
        assert result.read_text() == "gguf-bytes"
        assert not (adapter_dir / "adapter.gguf.converting").exists()


# ============================================================================
# manage.py — accepted-baseline pointer for the bench gate
# ============================================================================

GOOD_METRICS = {
    "total_cases": 100, "tool_selection_accuracy": 0.80,
    "task_completion_rate": 0.60, "format_compliance": 1.0,
    "hallucination_rate": 0.0, "canary_pass_rate": 0.90,
}
BAD_METRICS = dict(GOOD_METRICS, tool_selection_accuracy=0.50,
                   task_completion_rate=0.30)


class TestBaselinePointer:
    def _result(self, ft_home, name, metrics):
        import common
        path = common.BENCH_DIR / "results" / name
        _write_json(path, {"metrics": metrics, "cases": []})
        return path

    def test_first_run_passes_without_baseline(self, ft_home):
        from manage import bench_passes

        candidate = self._result(ft_home, "bench_1.json", GOOD_METRICS)
        passed, report = bench_passes(candidate)
        assert passed is True
        assert "new baseline" in report.lower()

    def test_gate_compares_against_accepted_baseline_only(self, ft_home):
        """A newer (even passing) bench_*.json file must NOT become the
        comparison point — only the explicit baseline.json is."""
        import manage

        good = self._result(ft_home, "bench_good.json", GOOD_METRICS)
        manage.update_baseline(good)
        assert manage.accepted_baseline_path().exists()

        # A newer result file that regressed — under the old mtime scan it
        # would have become the next run's baseline.
        bad = self._result(ft_home, "bench_bad.json", BAD_METRICS)
        passed, report = manage.bench_passes(bad)
        assert passed is False
        assert "baseline.json" in report

        # Even after the bad result exists, a fresh good candidate is still
        # judged against the ACCEPTED baseline (and passes).
        again = self._result(ft_home, "bench_good2.json", GOOD_METRICS)
        passed, report = manage.bench_passes(again)
        assert passed is True

    def test_regressed_run_never_updates_baseline(self, ft_home, monkeypatch):
        import manage

        good = self._result(ft_home, "bench_good.json", GOOD_METRICS)
        manage.update_baseline(good)
        before = manage.accepted_baseline_path().read_text()

        import common
        _write_json(common.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v1", "status": "previous"},
            {"cluster_id": "c1", "version": "v2", "status": "active",
             "rollback_target": "v1"},
        ]})
        for v in ("v1", "v2"):
            (common.ADAPTERS_DIR / "c1" / v).mkdir(parents=True)
        monkeypatch.setattr(manage, "redeploy", lambda *a, **k: True)
        monkeypatch.setattr(manage, "redeploy_base", lambda *a, **k: True)

        registry = manage.AdapterRegistry()
        bad = self._result(ft_home, "bench_bad.json", BAD_METRICS)
        assert manage.apply_bench_gate(
            registry, [("c1", "v2")], ("c1", "v2"), bad,
        ) is False

        # Baseline untouched; the regressed version rolled back.
        assert manage.accepted_baseline_path().read_text() == before
        statuses = {a["version"]: a["status"] for a in registry.registry["adapters"]}
        assert statuses == {"v1": "active", "v2": "rolled_back"}

    def test_passing_gate_updates_baseline(self, ft_home, monkeypatch):
        import common
        import manage

        _write_json(common.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v1", "status": "active"},
        ]})
        monkeypatch.setattr(manage, "redeploy", lambda *a, **k: True)
        monkeypatch.setattr(manage, "redeploy_base", lambda *a, **k: True)

        registry = manage.AdapterRegistry()
        candidate = self._result(ft_home, "bench_good.json", GOOD_METRICS)
        assert manage.apply_bench_gate(
            registry, [("c1", "v1")], ("c1", "v1"), candidate,
        ) is True

        data = json.loads(manage.accepted_baseline_path().read_text())
        assert data["source"] == "bench_good.json"
        assert data["metrics"] == GOOD_METRICS


# ============================================================================
# manage.py — regressing FIRST adapter is deactivated, base model re-served
# ============================================================================

class TestFirstAdapterDeactivation:
    def test_regressed_first_adapter_is_deactivated_not_reserved(
            self, ft_home, monkeypatch):
        import common
        import manage

        # First-ever adapter: active with no rollback target.
        _write_json(common.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v1", "status": "active",
             "rollback_target": None},
        ]})
        cluster_dir = common.ADAPTERS_DIR / "c1"
        (cluster_dir / "v1").mkdir(parents=True)
        (cluster_dir / "active").symlink_to("v1")

        good = common.BENCH_DIR / "results" / "bench_good.json"
        _write_json(good, {"metrics": GOOD_METRICS, "cases": []})
        manage.update_baseline(good)
        bad = common.BENCH_DIR / "results" / "bench_bad.json"
        _write_json(bad, {"metrics": BAD_METRICS, "cases": []})

        calls = []
        monkeypatch.setattr(manage, "redeploy",
                            lambda *a, **k: calls.append("adapter") or True)
        monkeypatch.setattr(manage, "redeploy_base",
                            lambda *a, **k: calls.append("base") or True)

        registry = manage.AdapterRegistry()
        assert manage.apply_bench_gate(
            registry, [("c1", "v1")], ("c1", "v1"), bad,
        ) is False

        entry = registry.registry["adapters"][0]
        assert entry["status"] == "deactivated"
        assert not (cluster_dir / "active").exists()
        # The regressed adapter is never re-served: base model only.
        assert calls == ["base"]
        # And the failure never became the ratchet.
        data = json.loads(manage.accepted_baseline_path().read_text())
        assert data["metrics"] == GOOD_METRICS


# ============================================================================
# manage.py / train.py — coarse pipeline lock
# ============================================================================

class TestPipelineLock:
    def test_second_invocation_fails_fast_with_clear_message(
            self, ft_home, monkeypatch, capsys):
        import common
        import manage

        monkeypatch.setattr(manage, "LOCK_TIMEOUT", 0.2)
        monkeypatch.setattr(sys, "argv", ["manage.py", "gc", "--keep", "1"])

        with common.pipeline_lock():
            with pytest.raises(SystemExit) as excinfo:
                manage.main()
        assert excinfo.value.code == 1
        assert "another finetune operation is running" in \
            capsys.readouterr().out.lower()

    def test_mutating_command_runs_once_lock_is_free(
            self, ft_home, monkeypatch, capsys):
        import manage

        monkeypatch.setattr(manage, "LOCK_TIMEOUT", 0.2)
        monkeypatch.setattr(sys, "argv", ["manage.py", "gc", "--keep", "1"])
        manage.main()
        assert "Garbage collection complete." in capsys.readouterr().out

    def test_status_needs_no_lock(self, ft_home, monkeypatch, capsys):
        import common
        import manage

        monkeypatch.setattr(manage, "LOCK_TIMEOUT", 0.2)
        monkeypatch.setattr(sys, "argv", ["manage.py", "status"])
        with common.pipeline_lock():
            manage.main()  # must not raise
        assert "FINETUNE PIPELINE STATUS" in capsys.readouterr().out


# ============================================================================
# manage.py — run --with-bench refuses to gate what it cannot measure
# ============================================================================

class TestWithBenchRequiresAutoRedeploy:
    def _run(self, monkeypatch, serving):
        import manage
        monkeypatch.setattr(manage, "load_config", lambda: {"serving": serving})
        with pytest.raises(SystemExit) as excinfo:
            manage.run_pipeline(with_bench=True)
        return excinfo.value.code

    def test_auto_redeploy_off_hard_fails_at_startup(
            self, ft_home, monkeypatch, capsys):
        code = self._run(monkeypatch, {
            "auto_redeploy": False, "server_command": "llama-server -m x",
        })
        assert code == 2
        out = capsys.readouterr().out
        assert "auto_redeploy" in out
        assert "PREVIOUSLY served" in out

    def test_missing_server_command_also_hard_fails(
            self, ft_home, monkeypatch, capsys):
        code = self._run(monkeypatch, {
            "auto_redeploy": True, "server_command": "",
        })
        assert code == 2
        assert "server_command" in capsys.readouterr().out


# ============================================================================
# plugin — manifest-gated, scale-based per-request routing
# ============================================================================

def _load_routing_plugin():
    """Import the plugin module fresh from its bundle location (fresh module
    each call so router/manifest caches never leak between tests)."""
    import importlib.util

    plugin_init = (
        Path(__file__).resolve().parent.parent
        / "optional-skills" / "mlops" / "finetune"
        / "plugin" / "finetune-routing" / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finetune_routing_plugin_lifecycle", plugin_init)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRoutingPluginServing:
    HEALTH_URL = "http://localhost:8008/v1/models"

    def _plugin(self, route_result, enabled=True):
        import common

        mod = _load_routing_plugin()
        mod._common_mod = common  # paths patched by ft_home

        class _Router:
            def __init__(self):
                self.enabled = enabled
                self.calls = []

            def route(self, prompt):
                self.calls.append(prompt)
                return route_result

        router = _Router()
        mod._router = router
        mod._router_failed = False
        return mod, router

    def _write_manifest(self, cluster="c-dev", health_url=None):
        import common
        _write_json(common.FINETUNE_DIR / "serving.json", {
            "updated_at": "2026-01-01T00:00:00",
            "server": {"pid": 1234,
                       "health_url": health_url or self.HEALTH_URL},
            "adapters": [{"id": 0, "cluster": cluster, "version": "v2",
                          "gguf": "/adapters/c-dev/v2/adapter.gguf"}],
        })

    def _request(self):
        return {
            "model": "carnice",
            "messages": [{"role": "user",
                          "content": "Refactor the config loader in this repo"}],
        }

    def test_matching_cluster_gets_scale_one(self, ft_home):
        mod, router = self._plugin({
            "cluster_id": "c-dev", "adapter_path": "/x.gguf",
            "confidence": 0.91, "label": "coding",
        })
        self._write_manifest(cluster="c-dev")

        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://localhost:8008/v1", model="carnice",
        )
        assert result is not None
        extra_body = result["request"]["extra_body"]
        # Real llama.cpp per-request field: preloaded adapter by index.
        assert extra_body["lora"] == [{"id": 0, "scale": 1.0}]
        assert "lora_adapters" not in extra_body  # nothing consumed that
        assert router.calls

    def test_off_domain_prompt_scales_adapters_to_zero(self, ft_home):
        """A prompt routed to an UNSERVED cluster must disable the served
        adapter (scale 0 → base model), not silently keep it applied."""
        mod, _ = self._plugin({
            "cluster_id": "c-writing", "adapter_path": None,
            "confidence": 0.7, "label": "writing",
        })
        self._write_manifest(cluster="c-dev")

        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://localhost:8008/v1", model="carnice",
        )
        assert result is not None
        assert result["request"]["extra_body"]["lora"] == [
            {"id": 0, "scale": 0.0}
        ]

    def test_inactive_without_manifest(self, ft_home):
        mod, router = self._plugin({
            "cluster_id": "c-dev", "adapter_path": "/x.gguf",
            "confidence": 0.91, "label": "coding",
        })
        # No serving.json written.
        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://localhost:8008/v1", model="carnice",
        )
        assert result is None
        assert not router.calls

    def test_host_lookalike_is_rejected(self, ft_home):
        """The old substring check matched 'notlocalhost.example.com' and
        leaked local routing onto remote APIs."""
        mod, router = self._plugin({
            "cluster_id": "c-dev", "adapter_path": "/x.gguf",
            "confidence": 0.91, "label": "coding",
        })
        self._write_manifest(cluster="c-dev")

        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="https://notlocalhost.example.com/v1", model="carnice",
        )
        assert result is None
        assert not router.calls

    def test_ipv6_loopback_alias_is_routed(self, ft_home):
        mod, _ = self._plugin({
            "cluster_id": "c-dev", "adapter_path": "/x.gguf",
            "confidence": 0.91, "label": "coding",
        })
        self._write_manifest(cluster="c-dev")

        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://[::1]:8008/v1", model="carnice",
        )
        assert result is not None

    def test_lan_host_matching_manifest_is_routed(self, ft_home):
        """Non-loopback llama.cpp (LAN box) routes when the request host
        matches the manifest's serving host exactly."""
        mod, _ = self._plugin({
            "cluster_id": "c-dev", "adapter_path": "/x.gguf",
            "confidence": 0.91, "label": "coding",
        })
        self._write_manifest(
            cluster="c-dev", health_url="http://192.168.1.50:8008/health")

        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://192.168.1.50:8008/v1", model="carnice",
        )
        assert result is not None
        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://192.168.1.51:8008/v1", model="carnice",
        )
        assert result is None

    def test_manifest_reload_on_mtime_change(self, ft_home):
        """A redeploy in another process (new manifest) must be visible
        without restarting the session."""
        import common

        mod, _ = self._plugin({
            "cluster_id": "c-dev", "adapter_path": "/x.gguf",
            "confidence": 0.91, "label": "coding",
        })
        self._write_manifest(cluster="c-dev")

        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://localhost:8008/v1", model="carnice",
        )
        assert result["request"]["extra_body"]["lora"][0]["scale"] == 1.0

        # Redeploy swaps the served adapter to another cluster.
        self._write_manifest(cluster="c-other")
        manifest = common.FINETUNE_DIR / "serving.json"
        bumped = manifest.stat().st_mtime_ns + 2_000_000_000
        os.utime(manifest, ns=(bumped, bumped))

        result = mod.finetune_llm_request_middleware(
            request=self._request(),
            base_url="http://localhost:8008/v1", model="carnice",
        )
        assert result["request"]["extra_body"]["lora"][0]["scale"] == 0.0


# ============================================================================
# route.py — state cache invalidation on mtime change
# ============================================================================

class TestRouteStateInvalidation:
    def test_rollback_in_other_process_is_visible(self, ft_home):
        import route as route_mod
        from route import AdapterRouter

        _write_json(route_mod.CLUSTER_STATE_PATH, {
            "centroids": {"c1": [1.0, 0.0]},
            "clusters": {"c1": {"label": "dev"}},
        })
        _write_json(route_mod.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v2", "status": "active"},
        ]})

        router = AdapterRouter(config={
            "clustering": {"confidence_threshold": 0.5},
            "routing": {"enabled": True},
        })
        assert router._get_active_adapters()["c1"]["version"] == "v2"

        # The pipeline rolls back in another process.
        _write_json(route_mod.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v2", "status": "rolled_back"},
            {"cluster_id": "c1", "version": "v1", "status": "active"},
        ]})
        bumped = route_mod.REGISTRY_PATH.stat().st_mtime_ns + 2_000_000_000
        os.utime(route_mod.REGISTRY_PATH, ns=(bumped, bumped))

        assert router._get_active_adapters()["c1"]["version"] == "v1"

    def test_routing_defaults_to_disabled(self):
        from route import AdapterRouter

        router = AdapterRouter(config={})  # no routing section at all
        assert router.enabled is False


# ============================================================================
# eval.py — empty gate-metric intersection fails closed
# ============================================================================

class TestVerdictEmptyIntersection:
    def test_empty_comparison_fails_closed(self, capsys):
        from eval import verdict

        checks = verdict({})
        assert checks["overall"] is False
        assert "failing closed" in capsys.readouterr().out

    def test_non_gate_metrics_only_fails_closed(self):
        """total_cases/mean_turns overlap alone gates nothing — that must
        FAIL, not vacuously pass."""
        from eval import compare_metrics, verdict

        baseline = {"total_cases": 100, "mean_turns": 3.0}
        candidate = {"total_cases": 100, "mean_turns": 2.0}
        checks = verdict(compare_metrics(candidate, baseline))
        assert checks["overall"] is False


# ============================================================================
# manage.py — cron wiring uses the real cron API and is honest on failure
# ============================================================================

class TestSetupCron:
    def _fake_cronjob(self, calls, jobs=None, success=True):
        def cronjob(action, **kwargs):
            calls.append((action, kwargs))
            if action == "list":
                return json.dumps({"success": True, "jobs": jobs or []})
            return json.dumps({
                "success": success,
                "job_id": "abc123",
                "message": "Cron job 'finetune-retrain' created.",
                "error": None if success else "boom",
            })
        return cronjob

    def test_creates_job_with_bench_gate_when_auto_redeploy(
            self, ft_home, monkeypatch, capsys):
        import manage

        calls = []
        monkeypatch.setattr(manage, "_import_cronjob_tool",
                            lambda: self._fake_cronjob(calls))
        monkeypatch.setattr(manage, "load_config", lambda: {
            "serving": {"auto_redeploy": True, "server_command": "x"},
        })

        assert manage.setup_cron("weekly") is True
        actions = [a for a, _ in calls]
        assert actions == ["list", "create"]
        _, kwargs = calls[1]
        assert kwargs["schedule"] == "0 3 * * 0"
        assert kwargs["name"] == "finetune-retrain"
        assert "--with-bench" in kwargs["prompt"]
        assert "created" in capsys.readouterr().out

    def test_ungated_prompt_is_honest_when_auto_redeploy_off(
            self, ft_home, monkeypatch, capsys):
        import manage

        calls = []
        monkeypatch.setattr(manage, "_import_cronjob_tool",
                            lambda: self._fake_cronjob(calls))
        monkeypatch.setattr(manage, "load_config", lambda: {
            "serving": {"auto_redeploy": False},
        })

        assert manage.setup_cron("daily") is True
        _, kwargs = calls[1]
        assert kwargs["schedule"] == "0 3 * * *"
        assert "--with-bench" not in kwargs["prompt"]
        assert "WITHOUT a benchmark gate" in kwargs["prompt"]
        assert "UNGATED" in capsys.readouterr().out

    def test_existing_job_is_updated_not_duplicated(
            self, ft_home, monkeypatch):
        import manage

        calls = []
        monkeypatch.setattr(
            manage, "_import_cronjob_tool",
            lambda: self._fake_cronjob(
                calls, jobs=[{"name": "finetune-retrain", "job_id": "j1"}]),
        )
        monkeypatch.setattr(manage, "load_config", lambda: {
            "serving": {"auto_redeploy": False},
        })

        assert manage.setup_cron("weekly") is True
        actions = [a for a, _ in calls]
        assert actions == ["list", "update"]
        _, kwargs = calls[1]
        assert kwargs["job_id"] == "j1"

    def test_unavailable_cron_api_is_honest_not_fake_success(
            self, ft_home, monkeypatch, capsys):
        """The old code claimed 'Cron job created' via an import that could
        never succeed. Unavailable tooling must say NO job was created and
        print the exact manual crontab line."""
        import manage

        monkeypatch.setattr(manage, "_import_cronjob_tool", lambda: None)
        monkeypatch.setattr(manage, "load_config", lambda: {
            "serving": {"auto_redeploy": False},
        })

        assert manage.setup_cron("weekly") is False
        out = capsys.readouterr().out
        assert "NO job was created" in out
        assert "0 3 * * 0" in out
        assert "manage.py run" in out

    def test_failed_create_reports_failure(self, ft_home, monkeypatch, capsys):
        import manage

        calls = []
        monkeypatch.setattr(
            manage, "_import_cronjob_tool",
            lambda: self._fake_cronjob(calls, success=False),
        )
        monkeypatch.setattr(manage, "load_config", lambda: {
            "serving": {"auto_redeploy": False},
        })

        assert manage.setup_cron("weekly") is False
        assert "boom" in capsys.readouterr().out


# ============================================================================
# manage.py — promote no-op + gc --keep 0
# ============================================================================

class TestRegistryMinorFixes:
    def test_repromote_active_version_is_noop(self, ft_home, capsys):
        import common
        from manage import AdapterRegistry

        _write_json(common.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v1", "status": "active",
             "rollback_target": None},
        ]})
        registry = AdapterRegistry()
        assert registry.promote("c1", "v1") is True
        entry = registry.registry["adapters"][0]
        # Must NOT point the rollback target at itself.
        assert entry["rollback_target"] is None
        assert entry["status"] == "active"
        assert "already active" in capsys.readouterr().out

    def test_gc_keep_zero_removes_all_unprotected(self, ft_home):
        import common
        from manage import AdapterRegistry

        cluster_dir = common.ADAPTERS_DIR / "c1"
        for v in ("v1", "v2", "v3"):
            (cluster_dir / v).mkdir(parents=True)
        _write_json(common.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v1", "status": "trained"},
            {"cluster_id": "c1", "version": "v2", "status": "trained"},
            {"cluster_id": "c1", "version": "v3", "status": "active"},
        ]})

        removed = AdapterRegistry().gc(keep_versions=0)
        assert sorted(removed) == ["c1/v1", "c1/v2"]
        assert (cluster_dir / "v3").exists()


# ============================================================================
# manage.py — deterministic redeploy target selection
# ============================================================================

class TestRedeployTargetSelection:
    def test_most_recently_promoted_active_adapter_wins(
            self, ft_home, monkeypatch, capsys):
        import common
        import manage

        # Two clusters active; c2 promoted later — must be chosen even
        # though c1 sorts first in the registry list.
        _write_json(common.REGISTRY_PATH, {"adapters": [
            {"cluster_id": "c1", "version": "v3", "status": "active",
             "promoted_at": "2026-01-01T00:00:00"},
            {"cluster_id": "c2", "version": "v1", "status": "active",
             "promoted_at": "2026-02-01T00:00:00"},
        ]})
        # No adapter_model dir anywhere → redeploy bails right after the
        # selection, which is all this test needs to observe.
        monkeypatch.setattr(manage, "load_config", lambda: {"serving": {}})

        assert manage.redeploy() is False
        out = capsys.readouterr().out
        assert "deploying c2 v1" in out
