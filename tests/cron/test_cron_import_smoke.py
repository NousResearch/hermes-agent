"""The cron dispatch import-chain smoke test catches a broken tree BEFORE 17 jobs fail.

Field incident (2026-10-24 02:16Z): ``agent/auxiliary_client.py`` gained a
``defaults=`` argument on the FUNCTIONAL ``NamedTuple`` form. ``typing.NamedTuple``
rejects that at IMPORT time ("Either list of fields or keywords can be provided
to NamedTuple, not both"). The cron external worker imports the LIVE checkout,
so 17 distinct cron jobs died in the same second, each reporting the same
opaque "exited before ownership acknowledgement (exit 1)".

These tests pin the three behaviours that turn that class of failure into ONE
actionable diagnostic:

* the probe names the module that failed and carries the traceback;
* the tick refuses to dispatch while the chain is broken (so the due jobs are
  left for a later tick instead of N identical failures);
* the per-job dispatch failure names the import error when the worker dies at
  import time (the case the tick probe cannot cover: the tree was edited
  between the probe and the spawn).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from cron import import_smoke


# The incident's own worker stderr, verbatim from the journal (2026-10-24
# 02:16:27Z). The chain is the one the worker walked.
INCIDENT_STDERR = """Traceback (most recent call last):
  File "/home/serveradmin/.hermes/hermes-agent/gateway/delivery.py", line 14, in <module>
    from .session import SessionSource
  File "/home/serveradmin/.hermes/hermes-agent/gateway/session.py", line 22, in <module>
    from gateway.session_transcript import SessionTranscriptMixin
  File "/home/serveradmin/.hermes/hermes-agent/gateway/session_transcript.py", line 11, in <module>
    from agent.turn_context import extract_api_content_sidecar
  File "/home/serveradmin/.hermes/hermes-agent/agent/turn_context.py", line 20, in <module>
    from agent.conversation_compression import recover_rotated_compression_session
  File "/home/serveradmin/.hermes/hermes-agent/agent/conversation_compression.py", line 31, in <module>
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
  File "/home/serveradmin/.hermes/hermes-agent/agent/auxiliary_client.py", line 7759, in <module>
    _LadderRoute = NamedTuple("_LadderRoute", [
        ("client", Any), ("task", Optional[str]), ("tag", str), ("async_mode", bool), ("base_info", str),
        ...<4 lines>...
        ("required_tokens", Optional[int]),
    ], defaults=(None,))
  File "/home/serveradmin/.hermes/tools/python-3.14.7/lib/python3.14/typing.py", line 3104, in NamedTuple
    raise TypeError("Either list of fields or keywords"
TypeError: Either list of fields or keywords can be provided to NamedTuple, not both
"""


# ---------------------------------------------------------------------------
# The chain definition
# ---------------------------------------------------------------------------

def test_chain_covers_the_incident_import_path():
    """Every module the incident's worker walked is in the probed chain."""
    for module in (
        "gateway.delivery",
        "gateway.session",
        "gateway.session_transcript",
        "agent.turn_context",
        "agent.conversation_compression",
        "agent.auxiliary_client",
    ):
        assert module in import_smoke.SMOKE_CHAIN


def test_chain_is_ordered_outermost_first():
    """The first module in the chain is the one the worker imports first."""
    assert import_smoke.SMOKE_CHAIN[0] == "gateway.delivery"


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def test_healthy_chain_reports_ok():
    """The real checkout passes the probe (the tree is not broken)."""
    result = import_smoke.check_import_chain(force=True)
    assert result.ok is True
    assert result.failing_module is None


def test_healthy_result_diagnostic_is_calm():
    result = import_smoke.SmokeResult(ok=True)
    assert "healthy" in result.diagnostic()


def test_broken_result_diagnostic_names_the_module_and_the_cause():
    result = import_smoke.SmokeResult(
        ok=False,
        failing_module="agent.auxiliary_client",
        error_type="TypeError",
        error_message="Either list of fields or keywords can be provided to NamedTuple, not both",
    )
    diagnostic = result.diagnostic()
    assert "agent.auxiliary_client" in diagnostic
    assert "TypeError" in diagnostic
    assert "Either list of fields or keywords" in diagnostic
    # The operator must learn that EVERY job on this path fails, not just one.
    assert "every job" in diagnostic.lower()


def test_probe_reports_the_failing_module_and_traceback(tmp_path, monkeypatch):
    """A broken module in the chain is named, and the traceback is carried."""
    broken = tmp_path / "broken_module.py"
    broken.write_text(
        "from typing import NamedTuple\n"
        "\n"
        "_T = NamedTuple('_T', [('a', int)], defaults=(None,))\n",
        encoding="utf-8",
    )
    result = import_smoke._run_probe(("broken_module",), tmp_path)
    assert result.ok is False
    assert result.failing_module == "broken_module"
    assert result.error_type == "TypeError"
    assert "NamedTuple" in (result.error_message or "")
    assert result.traceback
    assert "broken_module.py" in result.traceback


def test_probe_of_a_healthy_module_reports_ok(tmp_path):
    (tmp_path / "healthy_module.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = import_smoke._run_probe(("healthy_module",), tmp_path)
    assert result.ok is True


def test_probe_does_not_report_a_missing_third_party_dependency_as_a_broken_tree(tmp_path):
    """A missing external dependency is the WORKER's boot problem, not the tree's.

    The worker leases its own dependency generation; the tree can be perfectly
    fine. Failing the tick on a missing `ruamel` would turn a dependency
    problem into "the whole tree is broken", which is a different (and wrong)
    diagnosis.
    """
    (tmp_path / "needs_ruamel.py").write_text(
        "import ruamel.yaml\n", encoding="utf-8",
    )
    result = import_smoke._run_probe(("needs_ruamel",), tmp_path)
    assert result.ok is True


def test_probe_names_the_first_failing_module(tmp_path):
    (tmp_path / "first_bad.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    (tmp_path / "second_bad.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    result = import_smoke._run_probe(("first_bad", "second_bad"), tmp_path)
    assert result.failing_module == "first_bad"


def test_probe_names_the_CULPRIT_not_the_outermost_chain_member(tmp_path):
    """The module to FIX is named, not the first link in the walk.

    The 2026-10-24 incident's chain was
    ``gateway.delivery -> ... -> agent.auxiliary_client``. The first import
    that RAISED was ``gateway.delivery`` (it is the outermost link), but the
    file an operator must fix is ``agent/auxiliary_client``. Naming the
    outermost member would point the operator at the wrong file.
    """
    (tmp_path / "outer.py").write_text(
        "import broken_leaf\n", encoding="utf-8",
    )
    (tmp_path / "broken_leaf.py").write_text(
        "raise TypeError('boom')\n", encoding="utf-8",
    )
    result = import_smoke._run_probe(("outer",), tmp_path)
    assert result.ok is False
    # The leaf is where the code is bad; `outer` merely imported it.
    assert result.failing_module == "broken_leaf"
    assert result.error_type == "TypeError"


def test_probe_names_the_stdlib_raiser_not_the_tree_caller(tmp_path):
    """A stdlib frame that RAISED must not be named as the culprit.

    The incident's error was raised by ``typing.NamedTuple`` — naming that
    would send the operator to the standard library instead of to
    ``agent/auxiliary_client``.
    """
    (tmp_path / "caller.py").write_text(
        "import typing\n"
        "typing.NamedTuple('X', [('a', int)], defaults=(None,))\n",
        encoding="utf-8",
    )
    result = import_smoke._run_probe(("caller",), tmp_path)
    assert result.ok is False
    assert result.failing_module == "caller"
    assert result.error_type == "TypeError"


def test_probe_survives_a_probe_that_dies_without_a_verdict(tmp_path):
    """A probe that cannot even run still yields a usable verdict."""
    result = import_smoke._run_probe(("definitely_not_a_module_xyz",), tmp_path)
    assert result.ok is False
    assert result.failing_module == "definitely_not_a_module_xyz"


def test_cached_result_is_reused_for_an_unchanged_tree(monkeypatch):
    """An unchanged tree is probed once; the cached verdict is returned."""
    import_smoke.reset_cache()
    calls = []

    def _fake_probe(chain, repo_root):
        calls.append(chain)
        return import_smoke.SmokeResult(ok=True, chain=chain)

    monkeypatch.setattr(import_smoke, "_run_probe", _fake_probe)
    first = import_smoke.check_import_chain(use_cache=True)
    second = import_smoke.check_import_chain(use_cache=True)
    assert first.ok and second.ok
    assert len(calls) == 1


def test_cache_is_dropped_when_the_tree_changes(monkeypatch):
    """An edited tree is re-probed: the fix must be noticed without a restart."""
    import_smoke.reset_cache()
    fingerprints = iter(["aaa", "bbb"])

    monkeypatch.setattr(import_smoke, "_tree_fingerprint", lambda root: next(fingerprints))
    calls = []

    def _fake_probe(chain, repo_root):
        calls.append(chain)
        return import_smoke.SmokeResult(ok=True, chain=chain)

    monkeypatch.setattr(import_smoke, "_run_probe", _fake_probe)
    import_smoke.check_import_chain(use_cache=True)
    import_smoke.check_import_chain(use_cache=True)
    assert len(calls) == 2


def test_force_bypasses_the_cache(monkeypatch):
    import_smoke.reset_cache()
    calls = []

    def _fake_probe(chain, repo_root):
        calls.append(chain)
        return import_smoke.SmokeResult(ok=True, chain=chain)

    monkeypatch.setattr(import_smoke, "_run_probe", _fake_probe)
    import_smoke.check_import_chain(force=True)
    import_smoke.check_import_chain(force=True)
    assert len(calls) == 2


def test_reset_cache_drops_the_verdict():
    import_smoke.check_import_chain(force=True)
    import_smoke.reset_cache()
    assert import_smoke._cached_result is None
    assert import_smoke._cache_key is None


# ---------------------------------------------------------------------------
# The worker-stderr summariser
# ---------------------------------------------------------------------------

def test_worker_stderr_summariser_names_the_import_and_the_type():
    summary = import_smoke.summarize_worker_import_failure(INCIDENT_STDERR)
    assert summary is not None
    # The DEEPEST frame is the failure: auxiliary_client, not the
    # gateway/delivery.py that started the walk.
    assert "agent/auxiliary_client.py" in summary
    assert "TypeError" in summary
    assert "NamedTuple" in summary


def test_worker_stderr_summariser_ignores_a_non_import_failure():
    """A worker that died for another reason must not be reported as an import failure."""
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/x/y.py", line 3, in <module>\n'
        "    raise SystemExit(1)\n"
        "SystemExit: 1\n"
    )
    assert import_smoke.summarize_worker_import_failure(stderr) is None


def test_worker_stderr_summariser_ignores_empty_stderr():
    assert import_smoke.summarize_worker_import_failure("") is None


def test_worker_stderr_summariser_reports_the_exception_even_without_a_frame():
    stderr = "TypeError: something broke at import time\n"
    summary = import_smoke.summarize_worker_import_failure(stderr)
    assert summary is not None
    assert "TypeError" in summary
