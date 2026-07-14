"""Hermetic tests for scripts/mw/mw_doctor.py (MW-P3-I2f).

All probes are MOCK commands only — no real network, no real image APIs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# load module under test (path-stable; no package install required)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
MW_DOCTOR_PATH = REPO_ROOT / "scripts" / "mw" / "mw_doctor.py"

_spec = importlib.util.spec_from_file_location("mw_doctor", MW_DOCTOR_PATH)
assert _spec and _spec.loader
mw_doctor = importlib.util.module_from_spec(_spec)
sys.modules["mw_doctor"] = mw_doctor
_spec.loader.exec_module(mw_doctor)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _py_ok() -> List[str]:
    """Argv that exits 0 (mock tool ok)."""
    return [sys.executable, "-c", "import sys; sys.exit(0)"]


def _py_image_ok() -> List[str]:
    """Argv that exits 0 and prints a result line (proves a real image smoke)."""
    return [
        sys.executable,
        "-c",
        "print('image_smoke_ok result_id: mock'); import sys; sys.exit(0)",
    ]


def _py_fail() -> List[str]:
    """Argv that exits 1."""
    return [sys.executable, "-c", "import sys; sys.exit(1)"]


def _py_auth() -> List[str]:
    """Argv that prints 401 Unauthorized then exits 1."""
    return [
        sys.executable,
        "-c",
        "import sys; print('401 Unauthorized', file=sys.stderr); sys.exit(1)",
    ]


def _py_auth_exit0() -> List[str]:
    """Argv that prints 401 Unauthorized then exits 0 (false-ready trap)."""
    return [
        sys.executable,
        "-c",
        "import sys; print('401 Unauthorized'); sys.exit(0)",
    ]


def _py_relay_ok() -> List[str]:
    """Argv that prints relay JSON with status ok."""
    return [
        sys.executable,
        "-c",
        'import sys; print(\'{"status":"ok"}\'); sys.exit(0)',
    ]


def _py_relay_ok_then_exit1() -> List[str]:
    """Argv that prints ok JSON then exits 1 (crash after success text)."""
    return [
        sys.executable,
        "-c",
        'import sys; print(\'{"status":"ok"}\'); sys.exit(1)',
    ]


def _py_relay_auth() -> List[str]:
    """Argv that prints relay JSON with status auth."""
    return [
        sys.executable,
        "-c",
        'import sys; print(\'{"status":"auth"}\'); sys.exit(0)',
    ]


def _py_relay_non_json() -> List[str]:
    """Argv that prints non-JSON then exits 0."""
    return [
        sys.executable,
        "-c",
        "import sys; print('not json at all'); sys.exit(0)",
    ]


def _py_sleep(seconds: float) -> List[str]:
    """Argv that sleeps longer than a short timeout."""
    return [
        sys.executable,
        "-c",
        f"import time; time.sleep({seconds})",
    ]


def _missing_bin() -> List[str]:
    """Argv whose first element cannot be found."""
    return ["__mw_doctor_definitely_missing_binary_xyz__"]


def _config_yaml(
    *,
    tools: Optional[str] = None,
    images: Optional[str] = None,
    relay: Optional[str] = None,
    machine_note: Optional[str] = None,
) -> str:
    """Build a small mw-doctor config YAML string."""
    parts: List[str] = []
    if machine_note is not None:
        parts.append(f'machine_note: "{machine_note}"')
    if tools is not None:
        parts.append("tools:")
        parts.append(tools)
    if images is not None:
        parts.append("image_sources:")
        parts.append(images)
    if relay is not None:
        parts.append("relay:")
        parts.append(relay)
    return "\n".join(parts) + "\n"


def _yaml_quote(s: str) -> str:
    """Double-quoted YAML scalar with escapes (incl. newlines → \\n)."""
    esc = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{esc}"'


def _inline_list(argv: List[str]) -> str:
    """Format argv as a YAML inline list of quoted strings."""
    return "[" + ", ".join(_yaml_quote(a) for a in argv) + "]"


def _tool_entry(
    name: str,
    argv: List[str],
    *,
    optional: bool = False,
    expect_exit: int = 0,
) -> str:
    opt = "true" if optional else "false"
    # Nested under tools: → 2-space list item + 4-space keys
    return (
        f"  - name: {name}\n"
        f"    probe: {_inline_list(argv)}\n"
        f"    expect_exit: {expect_exit}\n"
        f"    optional: {opt}"
    )


def _image_entry(
    name: str,
    argv: Optional[List[str]],
    *,
    expect_exit: int = 0,
    expect_contains: Optional[str] = None,
    omit_smoke: bool = False,
) -> str:
    if omit_smoke:
        return f"  - name: {name}\n    expect_exit: {expect_exit}"
    assert argv is not None
    lines = [
        f"  - name: {name}",
        f"    smoke: {_inline_list(argv)}",
        f"    expect_exit: {expect_exit}",
    ]
    if expect_contains is not None:
        lines.append(f"    expect_contains: {_yaml_quote(expect_contains)}")
    else:
        lines.append("    expect_contains: null")
    return "\n".join(lines)


def _relay_entry(argv: List[str], expect_status: str = "ok") -> str:
    # Nested under relay: → keys at 2-space indent
    return (
        f"  smoke: {_inline_list(argv)}\n"
        f"  expect_status: {_yaml_quote(expect_status)}"
    )


def _write_config(tmp_path: Path, body: str, name: str = "mw-doctor.yaml") -> Path:
    path = tmp_path / name
    _write(path, body)
    return path


def _run_cli(config: Path, *extra: str) -> tuple:
    """Run mw_doctor.run() and capture stdout/stderr. Returns (code, out, err)."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    argv = ["--config", str(config), *extra]
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        code = mw_doctor.run(argv)
    return code, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# looks_like_auth_error helper
# ---------------------------------------------------------------------------

def test_looks_like_auth_error_markers():
    assert mw_doctor.looks_like_auth_error("401 Unauthorized")
    assert mw_doctor.looks_like_auth_error("missing API KEY for freepik")
    assert mw_doctor.looks_like_auth_error("credential not found")
    assert mw_doctor.looks_like_auth_error("HTTP 403 Forbidden")
    assert not mw_doctor.looks_like_auth_error("image saved to out.png")
    assert not mw_doctor.looks_like_auth_error("")


# ---------------------------------------------------------------------------
# READY path: all tools + images + relay ok → exit 0
# ---------------------------------------------------------------------------

def test_all_ok_ready_exit_0(tmp_path: Path):
    body = _config_yaml(
        machine_note="test-laptop",
        tools="\n".join(
            [
                _tool_entry("work_locks", _py_ok()),
                _tool_entry("helper", _py_ok()),
            ]
        ),
        images="\n".join(
            [
                _image_entry("freepik", _py_image_ok()),
                _image_entry("recraft", _py_image_ok()),
            ]
        ),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg)
    assert code == mw_doctor.EXIT_READY, (out, err)
    assert "mw-doctor: READY" in out
    assert "scope:" not in out  # full section=all is plain READY
    assert "work_locks OK" in out
    assert "freepik OK" in out
    assert "relay: OK" in out


def test_all_ok_json_shape(tmp_path: Path):
    body = _config_yaml(
        machine_note="ci-box",
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img1", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is True
    assert data["section"] == "all"
    assert data["blocking"] == []
    assert data["machine_note"] == "ci-box"
    assert "sections" in data
    assert len(data["sections"]["tools"]) == 1
    assert data["sections"]["tools"][0]["status"] == "ok"
    assert data["sections"]["tools"][0]["optional"] is False
    assert data["sections"]["images"][0]["status"] == "ok"
    assert data["sections"]["relay"]["status"] == "ok"


# ---------------------------------------------------------------------------
# required tool missing → NOT_READY
# ---------------------------------------------------------------------------

def test_required_tool_missing_not_ready(tmp_path: Path):
    body = _config_yaml(
        tools="\n".join(
            [
                _tool_entry("good", _py_ok()),
                _tool_entry("ghost", _missing_bin(), optional=False),
            ]
        ),
        images=_image_entry("freepik", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is False
    ghost = next(t for t in data["sections"]["tools"] if t["name"] == "ghost")
    assert ghost["status"] == "missing"
    assert any("ghost" in b for b in data["blocking"])


# ---------------------------------------------------------------------------
# optional tool missing → still READY
# ---------------------------------------------------------------------------

def test_optional_tool_missing_still_ready(tmp_path: Path):
    body = _config_yaml(
        tools="\n".join(
            [
                _tool_entry("work_locks", _py_ok()),
                _tool_entry("gitleaks", _missing_bin(), optional=True),
            ]
        ),
        images=_image_entry("freepik", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is True
    gitleaks = next(t for t in data["sections"]["tools"] if t["name"] == "gitleaks")
    assert gitleaks["status"] == "missing"
    assert gitleaks["optional"] is True
    assert data["blocking"] == []
    # human form marks optional
    code2, out2, _ = _run_cli(cfg)
    assert code2 == mw_doctor.EXIT_READY
    assert "MISSING(optional)" in out2 or "gitleaks" in out2


# ---------------------------------------------------------------------------
# image auth error → auth_missing, NOT_READY
# ---------------------------------------------------------------------------

def test_image_auth_error_auth_missing(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images="\n".join(
            [
                _image_entry("freepik", _py_image_ok()),
                _image_entry("topaz", _py_auth()),
            ]
        ),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    topaz = next(i for i in data["sections"]["images"] if i["name"] == "topaz")
    assert topaz["status"] == "auth_missing"
    assert data["ready"] is False
    assert any("topaz" in b and "auth_missing" in b for b in data["blocking"])


# ---------------------------------------------------------------------------
# image source with NO smoke → exit 2 config error
# ---------------------------------------------------------------------------

def test_image_no_smoke_config_error(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("freepik", None, omit_smoke=True),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg)
    assert code == mw_doctor.EXIT_ERR, (out, err)
    assert "no smoke command" in err.lower() or "auth-only" in err.lower()


# ---------------------------------------------------------------------------
# relay smoke JSON variants
# ---------------------------------------------------------------------------

def test_relay_status_ok(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_READY, (out, err)
    data = json.loads(out)
    assert data["sections"]["relay"]["status"] == "ok"


def test_relay_status_auth_not_ok(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_auth()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    assert data["sections"]["relay"]["status"] == "auth_missing"
    assert data["ready"] is False


def test_relay_non_json_not_ok(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_non_json()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    assert data["sections"]["relay"]["status"] == "fail"
    assert data["ready"] is False


def test_relay_ping_placeholder_substituted(tmp_path: Path):
    """PING token in smoke argv becomes a real temp file path."""
    # Helper script on disk (avoid multi-line -c inside YAML)
    helper = tmp_path / "relay_ping_check.py"
    helper.write_text(
        textwrap.dedent(
            """\
            import sys, json, pathlib
            found = False
            for a in sys.argv[1:]:
                p = pathlib.Path(a)
                if p.is_file():
                    text = p.read_text(encoding="utf-8")
                    if "mw-doctor ping" in text:
                        found = True
            if found:
                print(json.dumps({"status": "ok"}))
                sys.exit(0)
            print(json.dumps({"status": "fail", "reason": "no ping file"}))
            sys.exit(1)
            """
        ),
        encoding="utf-8",
    )
    argv = [sys.executable, str(helper), "PING"]
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(argv),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_READY, (out, err)
    data = json.loads(out)
    assert data["sections"]["relay"]["status"] == "ok"


# ---------------------------------------------------------------------------
# --skip-network + --section tools
# ---------------------------------------------------------------------------

def test_skip_network_images_and_relay_not_ready(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("freepik", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--skip-network", "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is False
    assert data["sections"]["tools"][0]["status"] == "ok"
    assert data["sections"]["images"][0]["status"] == "skipped(network)"
    assert data["sections"]["relay"]["status"] == "skipped(network)"
    # skipped is NOT ready
    assert any("skipped" in b for b in data["blocking"])


def test_section_tools_only_ready_when_tools_ok(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        # images would fail if run — but section tools ignores them
        images=_image_entry("broken", _py_auth()),
        relay=_relay_entry(_py_relay_non_json()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--section", "tools", "--json")
    assert code == mw_doctor.EXIT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is True
    assert data["section"] == "tools"
    assert "tools" in data["sections"]
    assert "images" not in data["sections"] or data["sections"].get("images") is None
    # section tools should not include images/relay evaluation results as blocking
    assert data["blocking"] == []
    # human form must qualify scoped READY (not full readiness)
    code_h, out_h, _ = _run_cli(cfg, "--section", "tools")
    assert code_h == mw_doctor.EXIT_READY
    assert "scope: tools" in out_h
    assert "mw-doctor: READY (scope: tools)" in out_h


def test_skip_network_section_tools_still_ok(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("freepik", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--section", "tools", "--skip-network", "--json")
    assert code == mw_doctor.EXIT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is True
    assert data["section"] == "tools"
    assert data["sections"]["tools"][0]["status"] == "ok"


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------

def test_timeout_probe_fails_not_hang(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("sleeper", _py_sleep(5.0)),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--timeout", "0.3", "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    sleeper = next(t for t in data["sections"]["tools"] if t["name"] == "sleeper")
    assert sleeper["status"] == "fail"
    assert "timeout" in sleeper["detail"].lower()


# ---------------------------------------------------------------------------
# config missing → exit 2
# ---------------------------------------------------------------------------

def test_config_missing_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    # no --config and no .work/mw-doctor.yaml
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        code = mw_doctor.run([])
    assert code == mw_doctor.EXIT_ERR
    assert "config not found" in err_buf.getvalue().lower() or "not found" in err_buf.getvalue().lower()


def test_default_config_path_under_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    default = tmp_path / ".work" / "mw-doctor.yaml"
    _write(default, body)
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        code = mw_doctor.run(["--json"])
    assert code == mw_doctor.EXIT_READY, (out_buf.getvalue(), err_buf.getvalue())
    data = json.loads(out_buf.getvalue())
    assert data["ready"] is True


# ---------------------------------------------------------------------------
# image expect_contains
# ---------------------------------------------------------------------------

def test_image_expect_contains_pass_and_fail(tmp_path: Path):
    has_token = [
        sys.executable,
        "-c",
        "print('result_id: abc123'); import sys; sys.exit(0)",
    ]
    body_ok = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("src", has_token, expect_contains="result_id"),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body_ok, "ok.yaml")
    code, out, _ = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_READY, out
    assert json.loads(out)["sections"]["images"][0]["status"] == "ok"

    body_bad = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("src", has_token, expect_contains="NOPE_NOT_HERE"),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg2 = _write_config(tmp_path, body_bad, "bad.yaml")
    code2, out2, _ = _run_cli(cfg2, "--json")
    assert code2 == mw_doctor.EXIT_NOT_READY
    assert json.loads(out2)["sections"]["images"][0]["status"] == "fail"


# ---------------------------------------------------------------------------
# required tool fail (exit 1) → NOT_READY
# ---------------------------------------------------------------------------

def test_required_tool_fail_exit(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("broken", _py_fail()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, _ = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_NOT_READY
    data = json.loads(out)
    assert data["sections"]["tools"][0]["status"] == "fail"
    assert data["ready"] is False


# ---------------------------------------------------------------------------
# mini-YAML loader can parse the schema (force mini)
# ---------------------------------------------------------------------------

def test_mini_yaml_loads_doctor_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    body = textwrap.dedent(
        """\
        tools:
          - name: work_locks
            probe: ["python3", "scripts/mw/work_locks.py", "--help"]
            expect_exit: 0
            optional: false
          - name: gitleaks
            probe: ["gitleaks", "version"]
            optional: true
        image_sources:
          - name: freepik
            smoke: ["freepik-cli", "search", "test", "--limit", "1"]
            expect_exit: 0
            expect_contains: null
        relay:
          smoke: ["relay-call", "--tool", "grok"]
          expect_status: "ok"
        """
    )
    data = mw_doctor.load_yaml_text(body, force_mini=True)
    assert isinstance(data, dict)
    assert len(data["tools"]) == 2
    assert data["tools"][0]["name"] == "work_locks"
    assert data["tools"][0]["probe"][0] == "python3"
    assert data["tools"][1]["optional"] is True
    assert data["image_sources"][0]["name"] == "freepik"
    assert data["image_sources"][0]["expect_contains"] is None
    assert data["relay"]["expect_status"] == "ok"


# ---------------------------------------------------------------------------
# extract_last_json_object
# ---------------------------------------------------------------------------

def test_extract_last_json_object():
    text = 'noise\n{"status":"fail"}\nmore\n{"status":"ok","n":1}\n'
    obj = mw_doctor.extract_last_json_object(text)
    assert obj is not None
    assert obj["status"] == "ok"
    assert mw_doctor.extract_last_json_object("no json here") is None


# ---------------------------------------------------------------------------
# live ready flag is derived (not hardcoded)
# ---------------------------------------------------------------------------

def test_ready_flag_live_from_blocking(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    raw = mw_doctor.load_yaml_file(cfg)
    report = mw_doctor.run_doctor(raw, section="all", skip_network=False, timeout=30)
    assert report.is_ready() is True
    assert report.to_json()["ready"] is True

    # mutate: inject a failing tool result and re-check live
    report.tools.append(
        mw_doctor.ItemResult(name="ghost", status="missing", optional=False)
    )
    assert report.is_ready() is False
    assert "ghost missing" in report.blocking()
    assert report.to_json()["ready"] is False


# ---------------------------------------------------------------------------
# human summary mentions blocking
# ---------------------------------------------------------------------------

def test_human_not_ready_lists_blocking(tmp_path: Path):
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("topaz", _py_auth()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, _ = _run_cli(cfg)
    assert code == mw_doctor.EXIT_NOT_READY
    assert "NOT_READY" in out
    assert "topaz" in out.lower() or "AUTH_MISSING" in out


# ---------------------------------------------------------------------------
# FIX ROUND 1 regressions (fail closed — no false READY)
# ---------------------------------------------------------------------------

def test_empty_config_section_all_exit_2(tmp_path: Path):
    """FIX 1: empty/missing sections must not READY with zero checks."""
    cfg = _write_config(tmp_path, "machine_note: empty-box\n")
    code, out, err = _run_cli(cfg)
    assert code == mw_doctor.EXIT_ERR, (out, err)
    assert "section all requires" in err.lower()
    assert "tools" in err.lower()
    assert "image_sources" in err.lower() or "relay" in err.lower()


def test_empty_tools_list_section_tools_exit_2(tmp_path: Path):
    """FIX 1: tools: [] + --section tools → exit 2 (nothing to prove)."""
    cfg = _write_config(tmp_path, "tools: []\n")
    code, out, err = _run_cli(cfg, "--section", "tools")
    assert code == mw_doctor.EXIT_ERR, (out, err)
    assert "section tools" in err.lower()
    assert "non-empty" in err.lower() or "requires" in err.lower()


def test_section_tools_scoped_ready_not_unqualified(tmp_path: Path):
    """FIX 2: scoped pass must report section + qualified READY text."""
    body = _config_yaml(tools=_tool_entry("t1", _py_ok()))
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--section", "tools", "--json")
    assert code == mw_doctor.EXIT_READY, (out, err)
    data = json.loads(out)
    assert data["section"] == "tools"
    assert data["ready"] is True
    code_h, out_h, _ = _run_cli(cfg, "--section", "tools")
    assert code_h == mw_doctor.EXIT_READY
    assert "scope: tools" in out_h
    assert "mw-doctor: READY (scope: tools)" in out_h
    # must not print bare full-readiness verdict for a partial section
    assert "mw-doctor: READY\n" not in out_h + "\n" or "scope:" in out_h


def test_image_empty_stdout_not_ok(tmp_path: Path):
    """FIX 3: exit 0 + empty stdout without expect_contains cannot prove a job."""
    # empty stdout smoke (same as bare sys.exit(0))
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("empty_src", _py_ok()),  # intentionally empty stdout
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body, "empty_out.yaml")
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is False
    img = data["sections"]["images"][0]
    assert img["status"] == "fail"
    assert "no result output" in img["detail"].lower() or "cannot prove" in img["detail"].lower()

    # printing a result line → ok
    body_ok = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("src", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg_ok = _write_config(tmp_path, body_ok, "with_out.yaml")
    code_ok, out_ok, _ = _run_cli(cfg_ok, "--json")
    assert code_ok == mw_doctor.EXIT_READY, out_ok
    assert json.loads(out_ok)["sections"]["images"][0]["status"] == "ok"

    # expect_contains set but not found → fail (already covered elsewhere; pin here)
    has_token = [
        sys.executable,
        "-c",
        "print('result_id: abc123'); import sys; sys.exit(0)",
    ]
    body_miss = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("src", has_token, expect_contains="NOPE_NOT_HERE"),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg_miss = _write_config(tmp_path, body_miss, "miss_token.yaml")
    code_miss, out_miss, _ = _run_cli(cfg_miss, "--json")
    assert code_miss == mw_doctor.EXIT_NOT_READY
    assert json.loads(out_miss)["sections"]["images"][0]["status"] == "fail"


def test_image_auth_on_exit_0_is_auth_missing(tmp_path: Path):
    """FIX 4: auth scan runs even when smoke exits 0."""
    body = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("topaz", _py_auth_exit0()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg = _write_config(tmp_path, body)
    code, out, err = _run_cli(cfg, "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is False
    topaz = data["sections"]["images"][0]
    assert topaz["status"] == "auth_missing"
    assert any("topaz" in b and "auth_missing" in b for b in data["blocking"])


def test_relay_exit_nonzero_not_ok_even_if_json_ok(tmp_path: Path):
    """FIX 5: relay must exit 0; ok JSON + exit 1 is NOT ready."""
    body_bad = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok_then_exit1()),
    )
    cfg_bad = _write_config(tmp_path, body_bad, "relay_crash.yaml")
    code, out, err = _run_cli(cfg_bad, "--json")
    assert code == mw_doctor.EXIT_NOT_READY, (out, err)
    data = json.loads(out)
    assert data["ready"] is False
    assert data["sections"]["relay"]["status"] == "fail"
    assert "exit" in data["sections"]["relay"]["detail"].lower()

    body_ok = _config_yaml(
        tools=_tool_entry("t1", _py_ok()),
        images=_image_entry("img", _py_image_ok()),
        relay=_relay_entry(_py_relay_ok()),
    )
    cfg_ok = _write_config(tmp_path, body_ok, "relay_ok.yaml")
    code_ok, out_ok, _ = _run_cli(cfg_ok, "--json")
    assert code_ok == mw_doctor.EXIT_READY, out_ok
    assert json.loads(out_ok)["sections"]["relay"]["status"] == "ok"
