"""Authenticated RoomLink orchestration of the lower operation-private selection."""
from contextlib import contextmanager, asynccontextmanager
import asyncio
import threading
from types import SimpleNamespace

from hermes_state_runtime import RuntimeStoreError


def room_route_scope(adapter, identity, request_identity, purpose):
    from gateway.session_peer_target import root_target
    from gateway.session_api import prospective_room_session
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.session_selected_route import selection_scope
    owner, _ = root_target(adapter, identity.target_profile)
    sid = prospective_room_session(owner, identity)
    source = SessionSource(platform=Platform.API_SERVER, chat_id=sid, user_id='api', chat_type='dm')
    return selection_scope(adapter, source=source,
        session_key=owner.runner.session_store._generate_session_key(source), session_id=sid,
        request_identity=request_identity, purpose=purpose)


def invitation_identity(identity, profile):
    return SimpleNamespace(**identity, target_profile=profile)


def current_room_route(adapter, *, session_id=None, purpose=None):
    from gateway.session_selected_route import held_selected_route
    binding = held_selected_route(adapter.gateway_runner)
    if binding is None:
        return None
    scope = binding._scope
    if (scope.adapter is not adapter or session_id is not None and scope.session_id != session_id
            or purpose is not None and scope.purpose != purpose):
        return None
    return binding


def room_route_ready(adapter, *, connection=None):
    from gateway.session_selected_route import peek_selected_route
    b = current_room_route(adapter)
    return bool(b is not None and peek_selected_route(b._scope, b, connection).supports_prepared_files)


def require_room_route(adapter, dispatch, *, connection=None):
    from gateway.session_api import hosted_session_id
    from gateway.session_selected_route import peek_selected_route
    b = current_room_route(adapter, session_id=hosted_session_id(dispatch))
    if b is None or not peek_selected_route(b._scope, b, connection).supports_prepared_files:
        raise RuntimeStoreError('prepared_files_unsupported')
    return b


def _prepare(scope, *, cancelled=None):
    from gateway.session_selected_route import prepare_selected_route, SelectedRouteUnavailable
    try:
        return prepare_selected_route(scope, cancelled=cancelled)
    except Exception:
        raise SelectedRouteUnavailable('selection_unavailable') from None


@contextmanager
def prepared_room_route(adapter, identity, request_identity, purpose, *, required=True, cancelled=None):
    from gateway.session_selected_route import prepare_selected_route, hold_selected_route
    scope = room_route_scope(adapter, identity, request_identity, purpose)
    with scope.runner._profile_scope_for_source(scope.source):
        b = _prepare(scope, cancelled=cancelled)
        try:
            with hold_selected_route(scope, b):
                if required and not room_route_ready(adapter):
                    raise RuntimeStoreError('prepared_files_unsupported')
                yield b
        finally:
            b.close()


@asynccontextmanager
async def prepared_room_route_async(adapter, identity, request_identity, purpose):
    """Credential selection off loop; cancelled completion cannot retain secrets."""
    from gateway.session_selected_route import prepare_selected_route, hold_selected_route
    scope = room_route_scope(adapter, identity, request_identity, purpose)
    cancelled = threading.Event()
    with scope.runner._profile_scope_for_source(scope.source):
        task = asyncio.create_task(asyncio.to_thread(_prepare, scope, cancelled=cancelled.is_set))
        try:
            b = await asyncio.shield(task)
        except BaseException:
            cancelled.set()
            def retire(done):
                if not done.cancelled() and done.exception() is None:
                    done.result().close()
            task.add_done_callback(retire)
            raise
        try:
            # Do not retain thread-owned locks across an asyncio suspension.
            # /runs takes the hold only around its synchronous commit section.
            yield b
        finally:
            b.close()


async def run_prepared_room_write(adapter, dispatch, request, purpose, write):
    """One worker owns selection and all writer holds; cancellation retires it."""
    cancelled = threading.Event()
    def run():
        with prepared_room_route(adapter, dispatch, request, purpose, cancelled=cancelled.is_set):
            if cancelled.is_set():
                raise RuntimeStoreError('selection_cancelled')
            return write()
    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        return await asyncio.shield(task)
    except BaseException:
        cancelled.set()
        # The worker owns its finally/secret cleanup even if the socket leaves.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        raise


def catalog_matches_dispatch(catalog, dispatch, *, admitted=False):
    """Only text tolerates a dynamic attachments-bit change, never other drift.

    Admitted Files validation reconstructs the already-authorized positive bit;
    fresh execution selection is separately required by the actual TurnRunner.
    """
    from gateway.hosted_room_peer import _catalog_digest
    if catalog['catalog_digest'] == dispatch.capability_digest:
        return True
    if dispatch.attachment_manifest_digest is not None:
        return admitted and _catalog_digest(dict(catalog, attachments=True)) == dispatch.capability_digest
    return _catalog_digest(dict(catalog, attachments=not catalog['attachments'])) == dispatch.capability_digest
