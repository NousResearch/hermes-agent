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
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from unittest.mock import patch


REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "optional-skills" / "creative" / "comfyui" / "scripts"


def _load_hardware_check():
    spec = spec_from_file_location("comfyui_hardware_check", SCRIPTS / "hardware_check.py")
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_nvidia_gpu_report_is_json_serializable():
    """The selected GPU must not create a circular `all_gpus` reference."""
    hardware_check = _load_hardware_check()
    nvidia_smi = "\n".join(
        [
            "0, NVIDIA RTX 3060, 12288, 576.52",
            "1, NVIDIA RTX 4090, 24576, 576.52",
        ]
    )

    with (
        patch.object(hardware_check.shutil, "which", return_value="nvidia-smi"),
        patch.object(hardware_check, "_run", return_value=nvidia_smi),
    ):
        gpu = hardware_check.detect_nvidia()

    assert gpu == {
        "vendor": "nvidia",
        "index": 1,
        "name": "NVIDIA RTX 4090",
        "vram_gb": 24.0,
        "driver": "576.52",
        "all_gpus": [
            {
                "vendor": "nvidia",
                "index": 0,
                "name": "NVIDIA RTX 3060",
                "vram_gb": 12.0,
                "driver": "576.52",
            },
            {
                "vendor": "nvidia",
                "index": 1,
                "name": "NVIDIA RTX 4090",
                "vram_gb": 24.0,
                "driver": "576.52",
            },
        ],
    }
    assert json.loads(json.dumps(gpu))["name"] == "NVIDIA RTX 4090"


def test_multi_rocm_gpu_report_is_json_serializable():
    """ROCm's sibling multi-GPU path has the same no-cycle contract."""
    hardware_check = _load_hardware_check()
    rocm_smi = json.dumps(
        {
            "card0": {
                "Card series": "Radeon RX 7900 XT",
                "VRAM Total Memory (B)": str(20 * 1024**3),
            },
            "card1": {
                "Card series": "Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": str(24 * 1024**3),
            },
        }
    )

    with (
        patch.object(hardware_check.shutil, "which", return_value="rocm-smi"),
        patch.object(hardware_check, "_run", return_value=rocm_smi),
    ):
        gpu = hardware_check.detect_rocm()

    assert gpu["name"] == "Radeon RX 7900 XTX"
    assert len(gpu["all_gpus"]) == 2
    assert json.loads(json.dumps(gpu))["vram_gb"] == 24.0

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
