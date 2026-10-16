"""Local divergence: continuing a historical messaging row adopts it as a local session (TUI-8, #134889).

Opt-in through ``gateway.session.viewer_replies_stay_local`` (default false). Upstream refuses a viewer's
turn on a messaging row its lane's route has moved past (``session_viewer_turn.require_current_lane``):
that turn would run in the lane's current session. Main's in-process TUI instead resumed any row under
``platform="tui"``. With the switch on, an operator's submission to such a row first binds it as the viewer's own local session through the legacy-adoption primitives
(``commit_row_adoption`` -> ``commit_local_session`` -> ``bind_legacy_target``), so the turn runs on a
local route of its own, answers in the panel and never touches the lane. The row's original identity
columns are kept in a preimage (``scripts/local_adopt_rollback.py``).

Gates of the legacy path this one does not apply, and why that is safe here:

* ``validate_legacy_row``'s refusal of routed messaging rows: see ``validate_viewer_row``; instead
  ``adopt_viewer_row`` requires that the lane's route (the gateway's routing index) names another session;
* ``adopt_legacy_session``'s registered native transport and ``session:create``: those prove ownership
  of an unowned transcript; this row was already mounted for the actor by ``authorize`` and the adoption
  creates no new session, so ``session:read`` (checked below) + ``session:submit`` (``submit``'s own
  authorization) are what it needs;
* ``legacy_account_label``: it vets a Desktop account stamp, not a platform sender.

``require_idle``, the turn lease and the active-session registry still apply (``bind_legacy_target``,
``commit_row_adoption``): they keep a second writer off the transcript.
"""
import logging
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


def adopt_moved_row(authority, actor, ref, payload):
    """Adopt *ref* for an operator submission (one carrying ``local_operator_v1``, written only by
    ``SessionAuthority._submit``) when its lane moved past it and replies stay local; else no-op.
    Runs under the mount's mutation lock, so of two racing submissions the second finds it local.
    Platform input, automation, cron, API and the lane's own sender never carry the envelope, and
    kanban, A2A and hosted rooms submit to local sessions (``lane_state`` None). An actor without
    ``session:read`` is not adopted: the submission gate refuses its turn visibly instead."""
    from gateway.session_viewer_turn import lane_state, viewer_replies_stay_local
    if ('local_operator_v1' not in payload or 'session:read' not in actor.capabilities
            or not viewer_replies_stay_local(authority) or lane_state(authority, ref) != 'moved'):
        return
    adopt_viewer_row(authority, ref.session_id, actor)


def adopt_viewer_row(authority, sid, actor):
    """Bind *sid* as *actor*'s local session and turn its messaging mount into the local one in place:
    the same ``LiveSession`` (subscribers, event stream, locks, pending prompts) gets the local source
    and route, so attached viewers and anything holding its locks see one continuous session."""
    from gateway.session_contract import SessionRef
    from gateway.session_local_migration import commit_row_adoption
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_viewer_turn import lane_state
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    # Only a row the lane has certainly moved past: one it still routes to is continued as the lane
    # (no second writer), and an unresolvable lane fails closed.
    if lane_state(authority, SessionRef(authority.profile_id, sid)) != 'moved':
        raise RuntimeStoreError('not_found')
    row = authority.db.get_session(sid)
    receipt = commit_row_adoption(authority, actor, row, 'tui', 'viewer-adopt:' + sid, viewer_adoption=True)
    # restore_local_session validates the receipt and publishes a fresh local mount; its source and
    # route move onto the existing mount. A failed publication leaves the row unmounted, and the next
    # read restores it from its local receipt.
    mount = authority.sessions.pop(sid)
    restore_local_session(authority, sid)
    local = authority.sessions[sid]
    mount.source, mount.route = local.source, local.route
    authority.sessions[sid] = mount
    logger.warning('Viewer adoption of messaging row %s: %s -> %s by %s (local divergence, #134889)',
                   sid, row['session_key'], receipt['route'], actor.subject)
