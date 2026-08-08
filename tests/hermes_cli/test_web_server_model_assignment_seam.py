"""Seam-identity and import-isolation regressions for the C9 extraction.

C9 (epic #78791) moved ``_normalize_main_model_assignment`` and
``_apply_main_model_assignment`` from ``hermes_cli/web_server.py`` into
``hermes_cli/web_server_model_assignment.py``.  ``web_server`` re-exports both
names at the original location; the identity invariant
``web_server.X is web_server_model_assignment.X`` must hold or the 8 call
sites and both dedicated unit files silently resolve to a stale definition.
"""

import subprocess
import sys
from pathlib import Path

import hermes_cli.web_server as web_server
import hermes_cli.web_server_model_assignment as web_server_model_assignment

SHIMMED_NAMES = ("_normalize_main_model_assignment", "_apply_main_model_assignment")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter rooted at the repo checkout."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_shim_identity_both_names():
    """T1 — the shim must expose the exact moved objects, not copies."""
    for name in SHIMMED_NAMES:
        assert getattr(web_server, name) is getattr(web_server_model_assignment, name)


def test_web_server_dict_still_exposes_both_names():
    """T1 — the re-export must land in web_server's own module namespace."""
    for name in SHIMMED_NAMES:
        assert name in web_server.__dict__


def test_standalone_import_pulls_no_web_server_or_fastapi():
    """T2 — fresh interpreter: standalone import, no heavy-dep pull."""
    code = (
        "import sys\n"
        "import hermes_cli.web_server_model_assignment as m\n"
        "assert 'hermes_cli.web_server' not in sys.modules\n"
        "assert 'fastapi' not in sys.modules\n"
        "assert 'starlette' not in sys.modules\n"
        "assert callable(m._normalize_main_model_assignment)\n"
        "assert callable(m._apply_main_model_assignment)\n"
        "print('STANDALONE-OK')\n"
    )
    proc = _run_fresh(code)
    assert proc.returncode == 0, proc.stderr
    assert "STANDALONE-OK" in proc.stdout


def test_import_order_web_server_first_then_module():
    """T9 — import-order fuzz, order A: web_server first."""
    code = (
        "import hermes_cli.web_server as ws\n"
        "import hermes_cli.web_server_model_assignment as m\n"
        "assert ws._normalize_main_model_assignment is m._normalize_main_model_assignment\n"
        "assert ws._apply_main_model_assignment is m._apply_main_model_assignment\n"
        "print('ORDER-A-OK')\n"
    )
    proc = _run_fresh(code)
    assert proc.returncode == 0, proc.stderr
    assert "ORDER-A-OK" in proc.stdout


def test_import_order_module_first_then_web_server():
    """T9 — import-order fuzz, order B: new module first."""
    code = (
        "import hermes_cli.web_server_model_assignment as m\n"
        "import hermes_cli.web_server as ws\n"
        "assert ws._normalize_main_model_assignment is m._normalize_main_model_assignment\n"
        "assert ws._apply_main_model_assignment is m._apply_main_model_assignment\n"
        "print('ORDER-B-OK')\n"
    )
    proc = _run_fresh(code)
    assert proc.returncode == 0, proc.stderr
    assert "ORDER-B-OK" in proc.stdout
