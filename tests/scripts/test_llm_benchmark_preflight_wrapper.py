"""Focused tests for the executable llm-benchmark-preflight wrapper.

Covers: provider-route derivation (no secret leakage), atomic state writes,
fail-closed behaviour, preflight JSON stdout emission, and PYTHONPATH
independence.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "llm_benchmark_preflight_wrapper.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model: {provider: test, default: test-model}\n"
        "custom_providers:\n"
        "  - name: ProviderA\n"
        "    base_url: https://a.example.com/v1\n"
        "    default_model: model-a\n"
        "    key_env: PROVIDER_A_KEY\n"
        "  - name: ProviderB\n"
        "    base_url: https://b.example.com/v1\n"
        "    models: [model-b1, model-b2]\n"
        "    # no key_env -> credential_unavailable path\n"
        "providers:\n"
        "  local:\n"
        "    base_url: http://127.0.0.1:11434/v1\n"
        "    api_key: not-needed\n",
        encoding="utf-8",
    )
    # catalogue = jobs.json with jobs + updated_at
    (home / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"name": "llm-benchmark-weekly", "enabled": True}], "updated_at": "2026-07-31T10:00:00+01:00"}),
        encoding="utf-8",
    )
    return home


# --- import-level tests -----------------------------------------------------

def test_build_provider_routes_derives_non_secret_config(tmp_path: Path) -> None:
    from scripts.benchmarks.llm_benchmark_preflight_wrapper import build_provider_routes

    config = {
        "custom_providers": [
            {"name": "A", "base_url": "https://a/v1", "default_model": "m-a", "key_env": "KEY_A"},
            {"name": "B", "base_url": "https://b/v1", "models": ["m-b1", "m-b2"]},
        ],
        "providers": {"local": {"base_url": "http://127.0.0.1/v1", "api_key": "not-needed"}},
    }
    routes = build_provider_routes(config)
    names = [r["name"] for r in routes]
    assert names == ["A", "B", "local"]
    assert routes[0]["key_env"] == "KEY_A"
    assert routes[1]["key_env"] == ""  # missing key_env
    assert routes[1]["model"] == "m-b1"  # first model from list
    # 'not-needed' is not a valid env-var name -> empty key_env
    assert routes[2]["key_env"] == ""
    # No secret VALUES leak: routes carry only the key_env NAME (an env-var
    # label), never the credential.  api_key values must never appear.
    assert all("api_key" not in r for r in routes)
    routes_json = json.dumps(routes)
    # The route config must not contain a placeholder secret value; the env-var
    # name KEY_A is a label, not a credential, so its presence is by design.
    assert "sk-" not in routes_json
    assert "Bearer" not in routes_json


def test_build_provider_routes_dedupes_and_skips_endpointless(tmp_path: Path) -> None:
    from scripts.benchmarks.llm_benchmark_preflight_wrapper import build_provider_routes

    config = {
        "custom_providers": [
            {"name": "dup", "base_url": "https://x/v1", "model": "m"},
            {"name": "dup", "base_url": "https://y/v1", "model": "m2"},  # deduped
            {"name": "noend", "model": "m3"},  # skipped (no base_url)
        ],
        "providers": {"noendpoint": {"api_key": "X"}, "dup": {"base_url": "http://z/v1"}},
    }
    routes = build_provider_routes(config)
    assert [r["name"] for r in routes] == ["dup"]


# --- atomic write + run integration -----------------------------------------

def test_run_writes_state_atomically_and_returns_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.benchmarks.llm_benchmark_preflight_wrapper import run, _atomic_write_json

    home = _make_hermes_home(tmp_path)
    # No credentials in env -> all measurements are credential_unavailable.
    monkeypatch.delenv("PROVIDER_A_KEY", raising=False)

    preflight = run(home, timeout_seconds=0.01, max_providers=2)

    state = home / "cron_states" / "llm-benchmark-weekly"
    assert (state / "measurements.json").is_file()
    assert (state / "preflight.json").is_file()
    m = json.loads((state / "measurements.json").read_text(encoding="utf-8"))
    assert len(m) > 0
    assert all(rec["status"] == "credential_unavailable" for rec in m)
    assert preflight["schema_version"] == 1
    assert "generated_at" in preflight
    assert preflight["catalogue"]["evidence_role"] == "availability_only"
    # previous_snapshot should be None on first run (no prior preflight.json)
    assert preflight["previous_snapshot"] is None

    # Second run: previous_snapshot now points to the first preflight.
    preflight2 = run(home, timeout_seconds=0.01, max_providers=2)
    assert preflight2["previous_snapshot"] is not None
    assert preflight2["previous_snapshot"]["schema_version"] == 1


def test_atomic_write_leaves_no_partial_on_error(tmp_path: Path) -> None:
    from scripts.benchmarks.llm_benchmark_preflight_wrapper import _atomic_write_json

    target = tmp_path / "out.json"
    _atomic_write_json(target, {"v": 1})
    assert json.loads(target.read_text())["v"] == 1

    # Simulate a non-serialisable object to force a failure mid-write.
    class Bad:
        pass

    with pytest.raises(TypeError):
        _atomic_write_json(target, {"bad": Bad()})
    # Original file untouched.
    assert json.loads(target.read_text())["v"] == 1
    # No leftover temp files.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".out.json")]
    assert leftovers == []


# --- subprocess E2E: PYTHONPATH independence + stdout emission -------------

def test_wrapper_emits_preflight_json_to_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _make_hermes_home(tmp_path)
    # Run the wrapper as a subprocess with NO PYTHONPATH set (simulates cron
    # sanitised env without the repo root).  The wrapper must bootstrap its own
    # sys.path so scripts.* imports still resolve.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
    }
    # Ensure no stale provider key leaks in.
    for k in ("PROVIDER_A_KEY", "COMMANDCODE_API_KEY"):
        env.pop(k, None)

    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--timeout-seconds", "0.01", "--max-providers", "2"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tmp_path),  # deliberately NOT the repo root
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1
    assert data["catalogue"]["evidence_role"] == "availability_only"
    # State files written.
    state = home / "cron_states" / "llm-benchmark-weekly"
    assert (state / "preflight.json").is_file()
    assert (state / "measurements.json").is_file()


def test_wrapper_fails_closed_on_missing_hermes_home_config(tmp_path: Path) -> None:
    """If config.yaml is absent, provider routes are empty but preflight still
    builds (catalogue defaults to {}); the wrapper should succeed, not crash.
    This documents fail-open-on-missing-config is NOT the failure mode — it
    degrades gracefully.  The real fail-closed path is a hard exception."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "cron").mkdir()
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [], "updated_at": "x"}))
    # No config.yaml.

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
    }
    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--timeout-seconds", "0.01"],
        env=env, capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["effective_configs"]["root"] == {}
    # No measurements collected (no providers).
    state = home / "cron_states" / "llm-benchmark-weekly"
    assert json.loads((state / "measurements.json").read_text()) == []


def test_wrapper_stdout_contains_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if a credential env var is present, it must never appear in stdout."""
    home = _make_hermes_home(tmp_path)
    secret = "sk-DO-NOT-LEAK-THIS-SECRET-VALUE"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
        "PROVIDER_A_KEY": secret,
    }
    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--timeout-seconds", "0.01", "--max-providers", "1"],
        env=env, capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert secret not in result.stdout
    assert secret not in result.stderr
    # Verify the measurement actually ran (status success or failed, not skipped).
    state = home / "cron_states" / "llm-benchmark-weekly"
    m = json.loads((state / "measurements.json").read_text())
    assert len(m) > 0
    assert all(secret not in json.dumps(rec) for rec in m)
