"""The session.create SOURCE SURFACE gate (build_policy) accepts the runtime's automation sources.

The full turn passes TWO source gates, and v0.22.2 only fixed the first one (t_862738cd):

1. ``tui_gateway/contracts/canonical_sessions.py`` — the canonical wire contract, enforced
   pre-handler by ``canonical_param_problems`` (``gateway/session_controls.py``). PR #141158
   reopened its ``source`` to ``str | None``.
2. ``gateway/session_policy.py`` — ``SURFACES`` (the create-label -> agent-surface map), enforced
   by ``build_policy`` at session-create time (``stage_local_session`` -> ``build_policy``),
   AFTER the wire gate clears. A source absent from ``SURFACES`` is refused here with
   ``invalid_params (source: not a surface: '<source>')``.

The kanban dispatcher injects ``HERMES_SESSION_SOURCE=kanban`` (hermes_cli/kanban_db_dispatch.py);
the worker's ``hermes chat -q`` ships it on ``session.create``. Gate 1 (the reopened contract)
accepted it, but gate 2 — the ``SURFACES`` map — refused it, so the kanban worker still failed
the full turn. The companion test ``test_canonical_create_kanban_source_gate.py`` only drives
``AuthorityConnection.dispatch`` and treats a handler exception as out of scope, so it CANNOT
catch a refusal this deep in the handler; it passed all 11 on the tree that still refused kanban
at gate 2. This test drives ``build_policy`` directly — the exact function the full turn calls —
and pins the accepted labels and their resulting platform.

The correct set mirrors ``restore_policy``: ``kanban`` runs as the CLI; ``cron`` and ``bot_room``
keep their own platform label. A session's ``source`` is a row LABEL, not a capability grant —
execution authz is the separate ``session:create`` capability / kanban claim — so the create gate
must accept exactly the labels the row store (and restore) already accept.
"""
import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from gateway.session_policy import SURFACES, build_policy, restore_policy

# The runtime's automation sources that a gateway-admitted session ships, and the platform each
# must resolve to (matching ``restore_policy``'s map). ``desktop`` is the Desktop app's row-store
# label (its canonical create actually ships ``source: 'gui'``), so it is intentionally NOT here:
# gate 2 is about the automation sources that reach the CLI worker, and desktop's wall was gate 1.
CREATED_SURFACES = {
    'kanban': 'cli',
    'cron': 'cron',
    'bot_room': 'bot_room',
    'cli': 'cli',
    'tool': 'cli',
    'oneshot': 'cli',
    'tui': 'tui',
    'gui': 'desktop',
    'acp': 'acp',
}


@pytest.mark.parametrize(('source', 'platform'), list(CREATED_SURFACES.items()))
def test_build_policy_builds_the_full_turn_for_every_runtime_source(tmp_path, source, platform):
    """The full turn's create step (``build_policy``) must ACCEPT each runtime source and stamp the
    agent surface ``restore_policy`` expects — the gate the dispatch-level regression test cannot
    reach. A refusal here is exactly the ``not a surface`` wall the kanban worker hit."""
    policy = build_policy({'source': source, 'cwd': str(tmp_path), 'model': 'm'}, {})
    assert policy.source == source
    assert policy.platform == platform


def test_build_policy_still_refuses_an_unknown_source(tmp_path):
    """The create gate stays a closed set of LABELS: an invented source is still refused (it is
    neither a UI transport nor an automation source the runtime stamps), so we did not open the
    field to arbitrary strings — only to the set the row store and ``restore_policy`` accept."""
    with pytest.raises(Exception) as exc:
        build_policy({'source': 'not_a_real_surface', 'cwd': str(tmp_path), 'model': 'm'}, {})
    # ``str()`` of a RuntimeStoreError is the bounded reason ('invalid_params'); the human detail
    # (``not a surface: ...``) rides in ``.detail``.
    assert getattr(exc.value, 'detail', None) == "not a surface: 'not_a_real_surface'"


def test_create_and_restore_share_one_surface_map():
    """Guard the v0.22.2 drift: the create gate (``SURFACES``) and the restore gate
    (``restore_policy``'s surface map) must name the same set, or a source one accepts and the
    other refuses is a restore loop (create refused 'kanban' while restore accepted it). If
    someone re-introduces an inline ``{**SURFACES, ...}`` in ``restore_policy`` to special-case a
    source the create gate lacks, this test fails at the seam instead of in the field."""
    src = textwrap.dedent(inspect.getsource(restore_policy))
    tree = ast.parse(src)
    inline = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'surfaces' for t in node.targets):
            inline = node.value
    # The map must be exactly ``SURFACES`` — not a dict-display built on top of it.
    assert inline is not None, 'restore_policy stopped assigning a local ``surfaces`` map'
    assert isinstance(inline, ast.Name) and inline.id == 'SURFACES', (
        f'restore_policy special-cases a source the create gate lacks: '
        f'expected bare ``SURFACES``, got {ast.unparse(inline)}')
