"""Tests for optional-skills/research/qmd-persistent-memory/scripts/qmd_memory_search.py.

Covers: JSON banner-stripping, qmd:// path normalization, mode routing
(vsearch vs query), collection scoping, and graceful error/timeout handling.
All subprocess calls are mocked — no live QMD, no network.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "research" / "qmd-persistent-memory" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import qmd_memory_search as qm  # noqa: E402


class _Ok:
    returncode = 0
    stdout = ""
    stderr = ""


class _Err(_Ok):
    returncode = 1
    stderr = "boom: qmd exploded"


def _qmd_present():
    """Context helper: patches QMD_BIN so _run_qmd reaches sp.run()."""
    return mock.patch.object(qm, "QMD_BIN", "/usr/local/bin/qmd")


def _run_ok(stdout):
    """Patches sp.run to return a successful result with given stdout.
    Does NOT patch QMD_BIN — caller must also apply _qmd_present()."""
    return mock.patch.object(qm.sp, "run", return_value=type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})())


def _rows():
    return [
        {"file": "qmd://HermesVault/02_Notes/x.md?index=memory&foo=1", "score": 0.91,
         "snippet": "We deferred the redis migration to Q3."},
        {"file": "qmd://wiki/agentforge/ok.md?index=memory", "score": 0.84,
         "snippet": "Decision recorded in the knowledge base."},
    ]


def _banner_rows():
    # QMD prints a banner line BEFORE the JSON payload.
    import json
    return "Expanding query...\nSearching 4 vector queries...\n" + json.dumps(_rows())


def test_fast_mode_routes_to_vsearch_and_strips_banner_and_scheme():
    with _qmd_present(), _run_ok(_banner_rows()) as sp, mock.patch.object(qm, "QMD_INDEX", "mem"):
        out = qm.semantic_search_memory("why did we defer it", n=5, mode="fast")
    args = sp.call_args.args[0]
    # _run_qmd builds: [QMD_BIN, "--index", QMD_INDEX, sub, query, "--json", "-n", n]
    assert args[1] == "--index" and args[2] == "mem"
    assert args[3] == "vsearch"
    assert args[-1] == "5"
    # banner stripped, concept match, qmd:// scheme + query-string removed
    assert "why did we defer it" not in out  # not echoing stdin back
    assert "HermesVault/02_Notes/x.md" in out
    assert "wiki/agentforge/ok.md" in out
    assert "?index=" not in out
    assert "0.91" in out


def test_hybrid_mode_routes_to_query():
    with _qmd_present(), _run_ok(_banner_rows()) as sp:
        qm.semantic_search_memory("hard lookup", n=3, mode="hybrid")
    args = sp.call_args.args[0]
    assert args[3] == "query"
    assert "hard lookup" in args


def test_default_mode_is_fast():
    with _qmd_present(), _run_ok(_banner_rows()) as sp:
        qm.semantic_search_memory("x", n=2)
    assert sp.call_args.args[0][3] == "vsearch"


def test_base_scoping_passes_collection_when_configured():
    qm.QMD_BASES["vault"] = "hermesvault"
    try:
        with _qmd_present(), _run_ok(_banner_rows()) as sp:
            qm.semantic_search_memory("x", n=2, base="vault")
        args = sp.call_args.args[0]
        assert "--collection" in args
        assert args[args.index("--collection") + 1] == "hermesvault"
        # "all" / unknown base -> no collection filter
        with _qmd_present(), _run_ok(_banner_rows()) as sp2:
            qm.semantic_search_memory("x", n=2, base="all")
        assert "--collection" not in sp2.call_args.args[0]
    finally:
        qm.QMD_BASES.pop("vault", None)


def test_missing_qmd_binary_returns_friendly_error():
    with mock.patch.object(qm, "QMD_BIN", None):
        out = qm.semantic_search_memory("x", n=2)
    assert "qmd not installed" in out


def test_nonzero_exit_returns_qmd_stderr():
    with _qmd_present(), mock.patch.object(qm.sp, "run", return_value=_Err()):
        out = qm.semantic_search_memory("x", n=2)
    assert "boom: qmd exploded" in out


def test_timeout_returns_actionable_message():
    def _timeout(*a, **k):
        raise qm.sp.TimeoutExpired("qmd", 1)

    with _qmd_present(), mock.patch.object(qm.sp, "run", side_effect=_timeout):
        out = qm.semantic_search_memory("x", n=2, mode="hybrid")
    assert "timed out" in out
    assert "mode='fast'" in out


def test_empty_results_return_no_match_message():
    with _qmd_present(), _run_ok("[]"):
        out = qm.semantic_search_memory("nothing relevant", n=3)
    assert "No relevant results" in out


def test_status_reports_ok():
    with _qmd_present(), _run_ok("QMD Status\nIndex: /x.sqlite\nSize: 14.5 MB\n"):
        out = qm.qmd_status()
    assert "QMD Status" in out