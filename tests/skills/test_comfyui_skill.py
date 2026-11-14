"""Invariant tests for the bundled comfyui skill.

Covers optional-skills/creative/comfyui — the diffusion workflow runner. Tests assert
contracts (locale-independent file reads), not snapshots of skill content.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "optional-skills" / "creative" / "comfyui" / "scripts"

# Text reads that must not depend on the host locale. The workflow and schema
# JSON are user-authored files (exported by ComfyUI or hand-edited), so they
# are read BOM-tolerantly: Notepad prepends U+FEFF, which makes json.load
# raise JSONDecodeError. See the jobs.json regression in #66607. The /proc
# reads are Linux-gated and never carry a BOM, so they pin plain utf-8.
_ENCODING_SENSITIVE_READS = [
    ("hardware_check.py", 'with open("/proc/version", "r", encoding="utf-8") as fh:'),
    ("hardware_check.py", 'with open("/proc/meminfo", "r", encoding="utf-8") as fh:'),
    ("run_workflow.py", 'with open(schema_path, encoding="utf-8-sig") as f:'),
    ("run_workflow.py", 'with wf_path.open(encoding="utf-8-sig") as f:'),
    # Sibling call paths: every other script that parses a user-authored
    # workflow JSON reads it the same BOM-tolerant way (same bug class).
    ("auto_fix_deps.py", 'wf_path.open(encoding="utf-8-sig")'),
    ("check_deps.py", 'wf_path.open(encoding="utf-8-sig")'),
    ("extract_schema.py", 'wf_path.open(encoding="utf-8-sig")'),
    ("health_check.py", 'wf_path.open(encoding="utf-8-sig")'),
    ("run_batch.py", 'wf_path.open(encoding="utf-8-sig")'),
]




def _run_under_c_locale(snippet: str) -> subprocess.CompletedProcess:
    """Execute a snippet in a child interpreter forced to a non-UTF-8 locale.

    The default text codec is resolved at interpreter startup, so the locale
    has to be set on the child's environment. Patching os.environ in-process
    would not change locale.getpreferredencoding(). PYTHONUTF8=0 disables
    PEP 540 UTF-8 mode, which would otherwise mask the bug entirely.
    """
    env = dict(os.environ)
    env.update({
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONIOENCODING": "utf-8",
    })
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_load_schema_reads_non_ascii_under_non_utf8_locale(tmp_path):
    """A schema with non-ASCII labels loads under the C locale.

    Without the explicit encoding the C-locale default codec is ASCII, so
    json.load crashes with UnicodeDecodeError on any CJK/Cyrillic label.
    """
    schema_path = tmp_path / "schema.json"
    schema_path.write_bytes(
        json.dumps(
            {"prompt": {"label": "プロンプト", "type": "string"}},
            ensure_ascii=False,
        ).encode("utf-8")
    )

    result = _run_under_c_locale(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(SCRIPTS)!r})
            from run_workflow import load_schema

            schema = load_schema({str(schema_path)!r}, {{}})
            assert schema["prompt"]["label"] == "\\u30d7\\u30ed\\u30f3\\u30d7\\u30c8", schema
            print("SUCCESS")
            """
        )
    )
    assert result.returncode == 0, (
        f"load_schema failed under non-UTF-8 locale:\n{result.stderr}"
    )
    assert "SUCCESS" in result.stdout


def test_load_schema_tolerates_utf8_bom(tmp_path):
    """A schema saved by a Windows GUI editor (UTF-8 BOM) still parses.

    json.load rejects a leading U+FEFF with JSONDecodeError, so a BOM-blind
    read turns "user edited the file in Notepad" into a hard failure.
    """
    schema_path = tmp_path / "schema.json"
    schema_path.write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"prompt": {"type": "string"}}).encode("utf-8")
    )

    result = _run_under_c_locale(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(SCRIPTS)!r})
            from run_workflow import load_schema

            schema = load_schema({str(schema_path)!r}, {{}})
            assert schema["prompt"]["type"] == "string", schema
            print("SUCCESS")
            """
        )
    )
    assert result.returncode == 0, (
        f"load_schema rejected a BOM-prefixed schema:\n{result.stderr}"
    )
    assert "SUCCESS" in result.stdout


def _load_auto_fix_deps():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import auto_fix_deps
    finally:
        sys.path.remove(str(SCRIPTS))
    return auto_fix_deps


def _record_subprocess(monkeypatch, module, stdout=""):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("env")))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return calls


def test_model_download_tokens_go_through_child_env_not_argv(monkeypatch, capsys):
    """Download tokens never reach comfy-cli argv (persisted --set-* setters, logged)."""
    mod = _load_auto_fix_deps()
    hf = "hf_" + "x" * 30
    civ = "civ" + "y" * 30
    calls = _record_subprocess(monkeypatch, mod, stdout=f"auth {hf} ok")

    assert mod.install_model("https://huggingface.co/o/r/resolve/main/m.safetensors",
                             "checkpoints", hf_token=hf, civitai_token=civ)
    assert mod.install_model("https://civitai.com/api/download/models/1",
                             "loras", hf_token=hf, civitai_token=civ)
    assert mod.install_model("https://huggingface.co.evil.example/m.safetensors",
                             "loras", hf_token=hf, civitai_token=civ)

    for cmd, _env in calls:
        assert not any(a.startswith("--set-") for a in cmd), cmd
        assert hf not in cmd and civ not in cmd, cmd
    (_, hf_env), (_, civ_env), (_, other_env) = calls
    assert hf_env["HF_API_TOKEN"] == hf and hf_env.get("CIVITAI_API_TOKEN") != civ
    assert civ_env["CIVITAI_API_TOKEN"] == civ and civ_env.get("HF_API_TOKEN") != hf
    assert other_env is None or (other_env.get("HF_API_TOKEN") != hf
                                 and other_env.get("CIVITAI_API_TOKEN") != civ)
    err = capsys.readouterr().err
    assert hf not in err and civ not in err


def _inherit_tokens(monkeypatch):
    hf = "hf_" + "i" * 30
    civ = "civ" + "j" * 30
    monkeypatch.setenv("HF_API_TOKEN", hf)
    monkeypatch.setenv("CIVITAI_API_TOKEN", civ)
    return hf, civ


def _child_env(env):
    # env=None means the child inherits os.environ unchanged.
    return dict(os.environ) if env is None else env


def test_node_install_gets_no_token_even_when_inherited(monkeypatch):
    """Node installs run third-party install code: no HF/CivitAI token, supplied or inherited."""
    _inherit_tokens(monkeypatch)
    mod = _load_auto_fix_deps()
    calls = _record_subprocess(monkeypatch, mod)
    assert mod.install_node("some-pack")
    (cmd, env), = calls
    env = _child_env(env)
    assert "some-pack" in cmd
    assert "HF_API_TOKEN" not in env and "CIVITAI_API_TOKEN" not in env
    assert env.get("PATH") == os.environ.get("PATH")


def test_inherited_tokens_are_scoped_to_matching_host(monkeypatch):
    hf, civ = _inherit_tokens(monkeypatch)
    mod = _load_auto_fix_deps()
    calls = _record_subprocess(monkeypatch, mod)
    for url in ("https://civitai.com/api/download/models/1",
                "https://huggingface.co/o/r/resolve/main/m.safetensors",
                "https://example.com/m.safetensors"):
        assert mod.install_model(url, "loras")
    civ_env, hf_env, other_env = (_child_env(env) for _cmd, env in calls)
    assert civ_env.get("CIVITAI_API_TOKEN") == civ and "HF_API_TOKEN" not in civ_env
    assert hf_env.get("HF_API_TOKEN") == hf and "CIVITAI_API_TOKEN" not in hf_env
    assert "HF_API_TOKEN" not in other_env and "CIVITAI_API_TOKEN" not in other_env


def test_echoed_inherited_token_is_redacted(monkeypatch, capsys):
    hf, _civ = _inherit_tokens(monkeypatch)
    mod = _load_auto_fix_deps()
    _record_subprocess(monkeypatch, mod, stdout=f"using token {hf}")
    seen = []
    real_run_cmd = mod.run_cmd
    monkeypatch.setattr(mod, "run_cmd",
                        lambda *a, **kw: seen.append(real_run_cmd(*a, **kw)) or seen[-1])
    assert mod.install_model("https://huggingface.co/o/r/resolve/main/m.safetensors", "loras")
    (_code, out), = seen
    assert hf not in out


def test_comfy_cli_host_aliases_get_their_token(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_API_TOKEN", raising=False)
    mod = _load_auto_fix_deps()
    hf = "hf_" + "a" * 30
    civ = "civ" + "b" * 30
    calls = _record_subprocess(monkeypatch, mod)
    for url in ("https://huggingface.com/o/r/m.safetensors", "https://hf.co/o/r/m.safetensors",
                "https://civitai.red/api/download/models/1"):
        assert mod.install_model(url, "loras", hf_token=hf, civitai_token=civ)
    envs = [_child_env(env) for _cmd, env in calls]
    assert [e.get("HF_API_TOKEN") for e in envs] == [hf, hf, None]
    assert [e.get("CIVITAI_API_TOKEN") for e in envs] == [None, None, civ]


def test_explicit_token_wins_and_dotless_lookalike_gets_none(monkeypatch):
    _inherit_tokens(monkeypatch)
    mod = _load_auto_fix_deps()
    hf = "hf_" + "e" * 30
    calls = _record_subprocess(monkeypatch, mod)
    for url in ("https://huggingface.co/o/r/m.safetensors", "https://evilhuggingface.co/m.safetensors"):
        assert mod.install_model(url, "loras", hf_token=hf)
    hf_env, lookalike_env = (_child_env(env) for _cmd, env in calls)
    assert hf_env.get("HF_API_TOKEN") == hf
    assert "HF_API_TOKEN" not in lookalike_env


def test_malformed_model_url_fails_closed_without_raising(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    mod = _load_auto_fix_deps()
    calls = _record_subprocess(monkeypatch, mod)
    bad_url = "https://" + "[::1/x"
    mod.install_model(bad_url, "loras", hf_token="hf_" + "c" * 30)
    (_cmd, env), = calls
    assert "HF_API_TOKEN" not in _child_env(env)


def test_run_cmd_redacts_supplied_token_from_output(monkeypatch):
    mod = _load_auto_fix_deps()
    secret = "hf_" + "z" * 30
    _record_subprocess(monkeypatch, mod, stdout=f"using token {secret}")
    _code, out = mod.run_cmd(["comfy"], env_overlay={"HF_API_TOKEN": secret})
    assert secret not in out
