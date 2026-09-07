"""Owner-only containment of one known Group Chat participant, not takeover."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway import hosted_rooms as rooms
from gateway.platforms.api_server_run_idempotency import GroupRunFreezeError
from gateway.platforms.api_server_run_scope import validate_room_run_scope

MAX_REQUEST_BYTES = 8 * 1024


def _error(message, code, status):
    from gateway.platforms.api_server import _openai_error

    return web.json_response(_openai_error(message, code=code), status=status)


def _authorize_owner(adapter, request):
    from gateway.platforms.api_server import _api_request_profile
    from hermes_constants import get_default_hermes_root, get_hermes_home

    if not adapter._expected_api_key():
        return adapter._auth_failed_response()
    denied = adapter._check_auth(request)
    if denied is not None:
        return denied
    if request.match_info.get("profile") or _api_request_profile.get() not in (None, "default"):
        return _error("Use this gateway's installation-owner connection.", "installation_owner_required", 403)
    try:
        root = get_default_hermes_root().expanduser().resolve()
        if get_hermes_home().expanduser().resolve() != root:
            return _error("Use this gateway's installation-owner connection.", "installation_owner_required", 403)
        store = adapter._run_idempotency_store
        if store.durable is not True or not isinstance(store._db_path, str):
            raise ValueError("durable owner store unavailable")
        if not all(callable(getattr(store, name, None)) for name in (
            "freeze_room_scope", "room_stop_snapshot", "is_scope_frozen", "group_control_open",
        )):
            raise ValueError("owner control is unsupported by this store")
        if Path(store._db_path).expanduser().resolve() != root / "runs_idempotency.db":
            raise ValueError("owner store does not belong to this root")
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
        return _error("Durable owner control is unavailable on this connection.", "group_stop_storage_unavailable", 503)
    return None


def _stop_local_records(adapter, snapshot):
    from gateway.platforms import api_server, api_server_runs

    scope = snapshot["scope"]
    for record in snapshot["runs"]:
        run_id = record["run_id"]
        if adapter._run_owners.get(run_id) != scope:
            continue
        agent = adapter._active_run_agents.get(run_id)
        task = adapter._active_run_tasks.get(run_id)
        if agent is None and task is None:
            continue
        status = adapter._run_statuses.get(run_id, {"status": record["status"]})
        api_server_runs._stop_loaded_run(adapter, run_id, status, agent, task, _api_server=api_server)
        api_server_runs._unregister_approval_notify(adapter._run_approval_sessions.get(run_id))


def _public_snapshot(adapter, snapshot):
    active, unresolved = 0, 0
    observed = []
    for record in snapshot["runs"]:
        run_id = record["run_id"]
        local = adapter._run_owners.get(run_id) == snapshot["scope"] and (
            run_id in adapter._active_run_tasks or run_id in adapter._active_run_agents)
        if local:
            active += 1
        else:
            unresolved += 1
        observed.append({"run_id": run_id, "status": record["status"], "locally_managed": local})
    unknown = snapshot.get("counts", {}).get("unknown", 0)
    state = "unresolved" if unresolved or unknown or snapshot["truncated"] else "stopping" if active else "no_active_recorded_runs"
    return {
        "object": "hermes.group_participant.stop", "command_id": snapshot["command_id"],
        "participant": snapshot["identity"], "admissions_frozen": True, "frozen_at": snapshot["frozen_at"],
        "work_state": state, "runs": observed, "truncated": snapshot["truncated"], "counts": snapshot["counts"],
        "coverage": "one_known_participant_in_this_runs_store",
        "limitations": ["other_participants_not_stopped", "unrecorded_work_not_certified",
                        "external_effects_not_undone", "no_unfreeze_or_takeover"],
    }


def http_routes(adapter):
    async def operate(request, *, stop):
        denied = _authorize_owner(adapter, request)
        if denied is not None:
            return denied
        try:
            store = adapter._run_idempotency_store
            if stop:
                body, denied = await adapter._read_json_body(request.clone(client_max_size=MAX_REQUEST_BYTES))
                if denied is not None:
                    return denied
                if set(body) != {"participant", "command_id", "confirm"} or body["confirm"] is not True:
                    return _error("Confirm the participant freeze explicitly.", "invalid_group_stop_request", 400)
                identity = validate_room_run_scope(body["participant"])
                if identity["target_install_id"] != rooms.local_authority_gateway_id():
                    return _error("This participant belongs to another gateway.", "group_stop_target_mismatch", 400)
                snapshot = await asyncio.to_thread(store.freeze_room_scope, identity, body["command_id"])
                # The admission barrier is committed before interrupt/reap work.
                _stop_local_records(adapter, snapshot)
                snapshot = await asyncio.to_thread(store.room_stop_snapshot, snapshot["command_id"])
            else:
                snapshot = await asyncio.to_thread(store.room_stop_snapshot, request.match_info["command_id"])
            return web.json_response(_public_snapshot(adapter, snapshot))
        except GroupRunFreezeError as exc:
            return _error(str(exc), exc.code, exc.status)
        except (TypeError, ValueError, KeyError):
            return _error("Invalid participant or command identity.", "invalid_group_stop_request", 400)
        except (OSError, RuntimeError, sqlite3.Error):
            return _error("Owner control could not be confirmed. Check this operation again.", "group_stop_unconfirmed", 503)

    async def stop(request):
        return await operate(request, stop=True)

    async def status(request):
        return await operate(request, stop=False)

    return [
        ("POST", "/v1/group-participants/stop", stop),
        ("GET", "/v1/group-participants/stop/{command_id}", status),
    ]
