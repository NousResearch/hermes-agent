"""
Lifecycle tests for the finetune pipeline review fixes.

Covers: eval gate metric intersection, exp-based perplexity, baseline
lookup, config generation guards, route artifact selection, bench-gate
stale-result rejection, gc retention policy, and llama-server
stop/start safety. Everything runs against tmp dirs and fake
executables — no real accelerate/docker/llama-server.
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
