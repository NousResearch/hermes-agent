"""Every AIAgent construction path must attribute its execution surface.

The shared-metrics contract normalises a missing or unrecognised ``platform``
to ``unknown`` / ``other``. That is the correct behaviour for a *bounded*
schema, but it means a construction site that forgets to declare its surface
is silently mis-attributed rather than loudly broken: fleet telemetry then
reports "unknown" for real, attributable traffic.

These are contract tests between two pieces of data -- the surfaces the
contract accepts, and the surface each construction path actually declares --
not snapshots of any current value.
"""

from __future__ import annotations

import pytest

from hermes_cli.observability import shared_metrics_contract as contract


def test_acp_editor_sessions_get_their_own_surface():
    """ACP (VS Code / Zed / JetBrains) is a real interactive surface, not 'other'.

    The ACP adapter declares ``platform="acp"``. If that value is not an
    accepted surface, the contract's closed-schema fallback buckets every
    editor session into ``other`` alongside genuinely unclassifiable traffic.
    """
    assert contract.execution_surface({"platform": "acp"}) == "acp"


def test_acp_sessions_are_interactive():
    """An editor session is a human at a keyboard, like cli/tui/desktop."""
    fields = contract.task_start_fields({"platform": "acp"})
    assert fields["entrypoint"] == "interactive"
    assert fields["execution_surface"] == "acp"


def test_batch_runs_declare_their_surface():
    """batch_runner builds agents from a fixed passthrough tuple.

    ``batch`` is already an accepted surface, so the only defect is that the
    runner never declares it -- every batch task run reports 'unknown'.
    """
    import batch_runner

    assert "platform" in batch_runner._AGENT_PASSTHROUGH, (
        "batch_runner._AGENT_PASSTHROUGH omits 'platform', so batch task runs are "
        f"attributed to {contract.execution_surface({})!r} despite 'batch' being a "
        "valid execution surface"
    )


@pytest.mark.parametrize(
    "platform",
    ["cli", "tui", "desktop", "batch", "acp", "api_server", "cron", "telegram", "subagent", "curator"],
)
def test_declared_platforms_resolve_to_a_named_surface(platform):
    """No production construction path should resolve to unknown/other.

    'unknown' must mean "this run genuinely could not be attributed", not
    "a construction site forgot to say who it was".
    """
    surface = contract.execution_surface({"platform": platform})
    assert surface not in {"unknown", "other"}, (
        f"platform={platform!r} resolves to {surface!r}; a real surface is being "
        "folded into the catch-all bucket"
    )


def test_subagent_forks_get_their_own_surface():
    """tools/delegate_tool.py builds every subagent child with platform="subagent".

    If that value is not an accepted surface, every subagent delegation task run
    is folded into "other" instead of being attributable as delegation traffic.
    """
    assert contract.execution_surface({"platform": "subagent"}) == "subagent"


def test_subagent_forks_are_delegated():
    """A subagent's task_start_fields must resolve to the "delegated" entrypoint.

    delegate_tool.py always sets parent_session_id on the fork, so task_entrypoint's
    own parent_session_id check resolves this first in practice -- this pins the
    _SURFACE_ENTRYPOINTS mapping too, so the classification does not silently depend
    on that incidental rescue.
    """
    fields = contract.task_start_fields({"platform": "subagent", "parent_session_id": "parent-1"})
    assert fields["entrypoint"] == "delegated"
    assert fields["execution_surface"] == "subagent"
    # Without parent_session_id (e.g. a bare/test construction), the mapping in
    # _SURFACE_ENTRYPOINTS is what carries the classification.
    assert contract.task_start_fields({"platform": "subagent"})["entrypoint"] == "delegated"


def test_curator_forks_get_their_own_surface():
    """agent/curator.py builds its review fork with platform="curator".

    curator forks carry no parent_session_id (they are not a delegated child of a
    live turn), so an undeclared surface here folds BOTH execution_surface and
    entrypoint into "other", losing curator activity from analytics entirely.
    """
    fields = contract.task_start_fields({"platform": "curator"})
    assert fields["execution_surface"] == "curator"
    assert fields["entrypoint"] == "background"


def test_unattributed_runs_still_report_unknown():
    """The catch-all must survive: a genuinely undeclared run is 'unknown'.

    This is the counterpart to the tests above -- fixing attribution must not
    be achieved by inventing a default that hides real gaps.
    """
    assert contract.execution_surface({}) == "unknown"
    assert contract.task_start_fields({})["entrypoint"] == "unknown"
