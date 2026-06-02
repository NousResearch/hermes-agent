import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "hermes_ops_smoke.py"


def _load_json(stdout: str) -> dict:
    return json.loads(stdout)


def test_ops_smoke_default_is_doctor_free_and_profile_scoped():
    cp = subprocess.run(
        [sys.executable, str(SCRIPT), "--skip-pytest"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert cp.returncode == 0
    report = _load_json(cp.stdout)
    names = [item["name"] for item in report["checks"]]
    assert "aux_config" in names
    assert "checkpoint_store" in names
    assert "git_diff_check" in names
    assert "doctor_default" not in names
    assert "doctor_redops" not in names
    assert "producers" not in cp.stdout


def test_ops_smoke_include_doctor_is_explicit(monkeypatch):
    # Unit-check the CLI gating without running the real doctor command.
    import importlib.util

    spec = importlib.util.spec_from_file_location("hermes_ops_smoke", SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    calls = []

    def fake_doctor(profile):
        calls.append(profile)
        return {"name": f"doctor_{profile}", "ok": True}

    monkeypatch.setattr(mod, "_check_aux_config", lambda: {"name": "aux_config", "ok": True})
    monkeypatch.setattr(mod, "_check_checkpoint_store", lambda: {"name": "checkpoint_store", "ok": True})
    monkeypatch.setattr(mod, "_git_diff_check", lambda: {"name": "git_diff_check", "ok": True})
    monkeypatch.setattr(mod, "_pytest_targeted", lambda: {"name": "pytest_targeted", "ok": True})
    monkeypatch.setattr(mod, "_doctor", fake_doctor)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--skip-pytest", "--include-doctor"])

    assert mod.main() == 0
    assert calls == ["default", "redops"]
