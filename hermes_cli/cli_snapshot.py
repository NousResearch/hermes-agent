"""Classic CLI snapshot restore: hand off the profile offline, then rejoin its authority."""
from __future__ import annotations

from contextlib import suppress
import logging
import os

from hermes_cli.cli_commands_session_tools import _pr, _t
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_profile_lifecycle import _confirmed, _failure

logger = logging.getLogger(__name__)


def restore_snapshot_command(cli, parts):
    from hermes_cli.backup import list_quick_snapshots, quick_snapshot_dir, restore_quick_snapshot
    from hermes_cli.gateway_tools_notify import gateway_store
    if len(parts) < 3:
        print(f"  {_t('snapshot.usage_restore')}")
        snaps = list_quick_snapshots(limit=1)
        if snaps:
            print(f"  {_t('snapshot.most_recent', snapshot_id=snaps[0]['id'])}")
        return
    snap_id = parts[2]
    try:
        idx = int(snap_id)
    except ValueError:
        idx = None
    if idx is not None:
        snaps = list_quick_snapshots()
        if not 1 <= idx <= len(snaps):
            return print(f"  {_t('snapshot.invalid_number', max=len(snaps))}")
        snap_id = snaps[idx - 1]["id"]
    store = gateway_store(cli)
    gateway = store is not None and getattr(cli, "_gateway_session_open", False)
    try:
        # An invalid/missing target must not take a working gateway offline.
        if gateway and quick_snapshot_dir(snap_id) is not None:
            restored = _restore_gateway_snapshot(cli, store, snap_id)
        else:
            local_db = getattr(cli, "_session_db", None)
            if local_db is not None and not gateway:
                with suppress(Exception):
                    local_db.close()
                    cli._session_db = None
            restored = restore_quick_snapshot(snap_id)
    except (GatewayClientError, OSError) as exc:
        logger.warning("Snapshot restore failed: %s", exc, exc_info=True)
        return _pr(f"  {_t('snapshot.restore_incomplete', snapshot_id=snap_id)}", f"  {exc}")
    if restored:
        print(f"  {_t('snapshot.restored', snapshot_id=snap_id)}")
        if not gateway:
            print(f"  {_t('snapshot.restart_recommended')}")
    elif snap_id in {s.get("id") for s in list_quick_snapshots(limit=10**6)}:
        print(f"  {_t('snapshot.restore_incomplete', snapshot_id=snap_id)}")
    else:
        print(f"  {_t('snapshot.not_found', snapshot_id=snap_id)}")


def _pause_owner(home, control_home, identity):
    from gateway.control_socket import pause_gateway_for_update, request_unserve_profile
    from hermes_cli.data_cleanup_holders import pause_acked
    satellite = control_home != home
    if satellite:
        ack = request_unserve_profile(control_home, home.name)
        accepted = _confirmed(ack, "unserved", home.name) or (
            isinstance(ack, dict) and ack.get("pending") and not ack.get("error"))
    else:
        accepted = pause_acked(pause_gateway_for_update(control_home, timeout=8), identity)
    if not accepted:
        # Stopping the host gateway would take down every profile it multiplexes.
        retry = (f"Park the profile with `hermes -p {home.name} gateway stop` and retry."
                 if satellite else "Stop the gateway and retry.")
        raise GatewayClientError(f"Gateway refused the snapshot maintenance handoff; nothing was restored. {retry}")


def _restore_gateway_snapshot(cli, store, snap_id):
    from gateway.control_socket import request_serve_profile_hot
    from hermes_cli.backup import restore_quick_snapshot
    from hermes_cli.data_cleanup_holders import manual_gateway_identity
    from hermes_cli.gateway_connection import GatewayConnection
    from hermes_cli.gateway_runtime import control_home_for, discover_gateway_endpoint
    from hermes_cli.profiles_remove import wait_profile_unowned
    from hermes_constants import get_hermes_home
    from hermes_state_registry import close_all_under
    if os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip():
        raise GatewayClientError("Snapshot restore needs the local gateway. Disconnect the remote gateway and retry.")
    home = get_hermes_home().resolve()
    observed = discover_gateway_endpoint(home)
    if observed.state != "ready" or observed.endpoint is None:
        raise GatewayClientError("Local gateway is not ready for snapshot maintenance. Reconnect and retry.")
    control_home = control_home_for(home, observed.endpoint).resolve()
    identity = None
    if control_home == home:
        # A supervised owner is relaunched right after the pause and takes the profile back.
        try:
            identity = manual_gateway_identity(home)
        except RuntimeError as exc:
            raise GatewayClientError(f"Nothing was restored: {exc}.") from exc
    store.close_reader()
    close_all_under(home)
    # A final viewer cannot redial from an idle poll while maintenance is replacing its database.
    store.connection.close()
    unserved = False
    try:
        _pause_owner(home, control_home, identity)
        unserved = control_home != home
        if not wait_profile_unowned(home):
            raise GatewayClientError("Gateway still owns the profile; nothing was restored. "
                                     "Wait for its running work to stop and retry.")
        return restore_quick_snapshot(snap_id)
    finally:
        if unserved:
            ack = request_serve_profile_hot(control_home, home.name)
            if not _confirmed(ack, "served", home.name):
                logger.warning("Profile re-serve after snapshot restore: %s", ack)
                print(f"  Profile '{home.name}' is not served again yet: {_failure(ack)}. "
                      "The host retries on its next rescan (within 30s).")
        store.snapshots.clear()
        store.current = None
        # The restore's own result or error decides the reply; a failed rejoin must not replace it.
        try:
            store.connection = GatewayConnection.shared()
            _rejoin_cli(cli, store)
        except Exception as exc:
            logger.warning("Rejoin after snapshot restore failed: %s", exc, exc_info=True)
            print(f"  Reconnecting to the gateway failed: {exc}. Run /new or restart the CLI.")


def _rejoin_cli(cli, store):
    from hermes_cli.cli_init_mixin import _ROUTE_FLAGS, _gateway_launch, lease_handed_over
    from hermes_cli.gateway_tools import bind_session
    # Same-id restores still replace the connection and invalidate the cached proxies.
    cli.agent = None
    for attr in ("_goal_manager", "_loop_manager", "_heartbeat_manager"):
        setattr(cli, attr, None)
    # A snapshot older than the gateway's import of a legacy row makes this attach import it again.
    with lease_handed_over(cli, getattr(cli, "_active_session_lease", None) is not None):
        try:
            snapshot = store.attach(cli.session_id)
        except GatewayClientError as exc:
            if str(exc) != "not_found":
                raise
            # The snapshot predates this session: only the owner may mint its replacement.
            launch = {key: value for key, value in _gateway_launch(cli).items() if key not in _ROUTE_FLAGS}
            store.create(launch, record_cwd=False)
            snapshot = store.current
        cli.session_id = snapshot["stored_session_id"]
        cli.conversation_history = [{key: value for key, value in message.items() if key != "row_id"}
                                    for message in snapshot.get("messages") or []]
        bind_session(cli, store.connection)
    getattr(cli, "_write_terminal_breadcrumb", lambda: None)()
