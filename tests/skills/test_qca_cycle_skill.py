"""
Tests for the qca-cycle cognitive skill.

Covers the review findings on PR #43306:
  - SKILL.md frontmatter conforms to the hardline format (description ≤ 60)
  - store paths resolve under the active Hermes home, not a hardcoded ~/.hermes
  - `pulse` honors the silent-cron contract (deliberate silence = empty stdout)
  - `ses_bridge.py verify` fails (non-zero exit) on tampering AND on the
    SES v5.1 canon-lock violation (STATE_SNAPSHOT without kernel_ref/kernel_hash)

All LLM calls use the mock backend; embeddings use the lexical fallback by
pointing OLLAMA_URL at a closed local port (no live network calls).
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "cognitive" / "qca-cycle"
ENGINE = SKILL_DIR / "scripts" / "qca_engine.py"
BRIDGE = SKILL_DIR / "scripts" / "ses_bridge.py"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def _env(tmp_home: Path, **extra: str) -> dict:
    """Hermetic env: Hermes home in tmp, mock LLM, embeddings in lexical
    fallback (OLLAMA_URL points at a closed port; refused instantly)."""
    env = os.environ.copy()
    for k in ("QCA_STORE", "QCA_KERNEL", "QCA_LEARNED_DIR", "ANTHROPIC_API_KEY"):
        env.pop(k, None)
    env.update({"HERMES_HOME": str(tmp_home),
                "QCA_LLM_BACKEND": "mock",
                "OLLAMA_URL": "http://127.0.0.1:1"})
    env.update(extra)
    return env


def _run(script: Path, *args: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, env=env, timeout=120)


# ── SKILL.md frontmatter (hardline authoring standards) ──────────────────

def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "qca-cycle"


def test_author_credits_contributor(frontmatter) -> None:
    assert "Trubnikov" in frontmatter["author"]


def test_platforms_cross_platform(frontmatter) -> None:
    # Scripts are pure stdlib with no POSIX-only primitives.
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


# ── Scripts: static checks ────────────────────────────────────────────────

@pytest.mark.parametrize("script", [ENGINE, BRIDGE,
                                    SKILL_DIR / "scripts" / "_hermes_home.py"])
def test_scripts_parse(script: Path) -> None:
    ast.parse(script.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script", [ENGINE, BRIDGE])
def test_no_hardcoded_hermes_home(script: Path) -> None:
    # Paths must resolve via HERMES_HOME / hermes_constants (profile isolation).
    assert "~/.hermes" not in script.read_text(encoding="utf-8")


# ── Store path resolution ─────────────────────────────────────────────────

def test_store_resolves_under_hermes_home(tmp_path: Path) -> None:
    r = _run(ENGINE, "stats", env=_env(tmp_path))
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout)
    assert stats["store"] == str(tmp_path / "qca" / "graph.json")
    assert (tmp_path / "qca" / "graph.json").is_file()


def test_qca_store_override_wins(tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere" / "graph.json"
    r = _run(ENGINE, "stats", env=_env(tmp_path, QCA_STORE=str(custom)))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["store"] == str(custom)


# ── pulse: silent-cron contract ───────────────────────────────────────────

def test_pulse_silence_is_empty_stdout(tmp_path: Path) -> None:
    env = _env(tmp_path)
    # First pulse on an empty graph: the (deterministic) mock thought is novel,
    # gets written and printed.
    first = _run(ENGINE, "pulse", env=env)
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip(), "first pulse should produce a thought"
    # Second pulse produces the identical mock thought; the novelty gate
    # discards it and the CLI must emit NOTHING on stdout (no 'SILENCE').
    second = _run(ENGINE, "pulse", env=env)
    assert second.returncode == 0, second.stderr
    assert second.stdout == "", f"silent pulse must print nothing, got: {second.stdout!r}"


# ── SES snapshots: export + verify ────────────────────────────────────────

def _export(tmp_home: Path, snap: Path) -> subprocess.CompletedProcess:
    return _run(ENGINE, "export-ses", str(snap), env=_env(tmp_home))


def test_verify_roundtrip_with_kernel(tmp_path: Path) -> None:
    (tmp_path / "qca").mkdir(parents=True)
    shutil.copy(SKILL_DIR / "kernel.example.ses.json", tmp_path / "qca" / "kernel.ses.json")
    snap = tmp_path / "snap.ses.json"
    r = _export(tmp_path, snap)
    assert r.returncode == 0, r.stderr
    doc = json.loads(snap.read_text(encoding="utf-8"))
    assert doc["snapshot_type"] == "COMBINED"
    assert doc["meta"]["kernel_hash"].startswith("sha256:")
    v = _run(BRIDGE, "verify", str(snap), env=_env(tmp_path))
    assert v.returncode == 0, v.stdout + v.stderr
    assert "integrity verified" in v.stdout


def test_verify_fails_on_tampering(tmp_path: Path) -> None:
    (tmp_path / "qca").mkdir(parents=True)
    shutil.copy(SKILL_DIR / "kernel.example.ses.json", tmp_path / "qca" / "kernel.ses.json")
    snap = tmp_path / "snap.ses.json"
    assert _export(tmp_path, snap).returncode == 0
    doc = json.loads(snap.read_text(encoding="utf-8"))
    doc["state"]["meta"]["summary"] = "tampered"
    snap.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    v = _run(BRIDGE, "verify", str(snap), env=_env(tmp_path))
    assert v.returncode == 1
    assert "TAMPERED" in v.stdout


def test_verify_fails_canon_violation(tmp_path: Path) -> None:
    """A STATE_SNAPSHOT without kernel_ref/kernel_hash violates the SES v5.1
    canon lock (§12) and must exit non-zero even when the hash matches."""
    snap = tmp_path / "snap.ses.json"
    r = _export(tmp_path, snap)  # no kernel installed
    assert r.returncode == 0
    assert "canon-lock" in r.stderr  # export warns about the degraded snapshot
    doc = json.loads(snap.read_text(encoding="utf-8"))
    assert doc["snapshot_type"] == "STATE_SNAPSHOT"
    assert "kernel_ref" not in doc["meta"] and "kernel_hash" not in doc["meta"]
    v = _run(BRIDGE, "verify", str(snap), env=_env(tmp_path))
    assert v.returncode == 1, "canon violation must fail verification"
    assert "canon violation" in v.stdout


# ── _hermes_home fallback resolver ────────────────────────────────────────

def test_hermes_home_fallback(monkeypatch, tmp_path: Path) -> None:
    """The stdlib fallback (used when hermes_constants is not importable)
    honors HERMES_HOME and the platform-native default."""
    import importlib.util
    monkeypatch.setitem(sys.modules, "hermes_constants", None)  # force ImportError
    spec = importlib.util.spec_from_file_location(
        "qca_hermes_home", SKILL_DIR / "scripts" / "_hermes_home.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert mod.get_hermes_home() == tmp_path

    monkeypatch.delenv("HERMES_HOME")
    if sys.platform == "win32":
        assert mod.get_hermes_home().name == "hermes"
    else:
        assert mod.get_hermes_home() == Path.home() / ".hermes"
