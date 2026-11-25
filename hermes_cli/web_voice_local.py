"""GPT-Live capture on the standalone serve process's TUI session store."""

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path

from gateway.owner_identity import same_owner
from gateway.session_contract import SessionActor
from gateway.session_persistence import owner_rule
from gateway.session_voice_capture import VoiceCaptures
from persistence import ProfileScope, SessionId
from persistence.authority.vocabulary import LEGACY_LOCAL_SOURCES
from persistence.conversation.sessions import SessionRecord
from persistence.core.errors import AuthorityRefusal, PersistenceError
from persistence_assembly import Persistence, open_persistence, open_persistence_reader, profile_scope, store_location
from session_rules.legacy_local import validate_legacy_row

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalVoiceChat:
    scope: ProfileScope
    session_id: SessionId
    draft: dict | None = None


def _is_legacy_row(store: Persistence, row: SessionRecord) -> bool:
    if row.source not in LEGACY_LOCAL_SOURCES or row.session_key or row.chat_id or row.origin:
        return False
    marks = store.bindings.binding_marks(row.session_id)
    return not (marks.policy or marks.legacy)


def _local_draft(scope: ProfileScope, session_id: SessionId) -> dict | None:
    from tui_gateway import server

    runtime_home = None if scope.profile_dir == server._launch_home().resolve() else scope.profile_dir
    live = server._find_live_session_by_key(session_id, runtime_home)
    if live is None:
        return None
    _, draft = live
    if draft.get('session_key') == session_id and server._session_source(draft) in LEGACY_LOCAL_SOURCES:
        return draft
    return None


def find_local_voice_chat(actor: SessionActor, session_id: str) -> LocalVoiceChat | None:
    """Decide ownership without opening the writer or creating an unknown chat."""
    from tui_gateway import server

    if 'session:control' not in actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    scope = profile_scope(home=Path(actor.profile_id))
    key = SessionId(session_id)
    path = store_location(scope)
    try:
        if path is not None and path.is_file():
            with open_persistence_reader(path) as store:
                row = store.sessions.get(key)
                if row is not None:
                    if not _is_legacy_row(store, row):
                        return None
                    validate_legacy_row(scope.name, row.as_session_row(), subject=actor.subject, rule=owner_rule())
                    return LocalVoiceChat(scope, key)
    except AuthorityRefusal:
        raise
    except (OSError, PersistenceError, ValueError) as exc:
        log.debug('GPT-Live local chat lookup deferred to the owner (%s)', type(exc).__name__)
        return None

    draft = _local_draft(scope, key)
    if draft is None:
        return None
    user = server._session_auth_user_id(draft)
    if user and not same_owner(user, actor.subject):
        raise AuthorityRefusal('not_found')
    return LocalVoiceChat(scope, key, draft)


def _prepare_capture(store: Persistence, chat: LocalVoiceChat, actor: SessionActor) -> SessionId | None:
    from tui_gateway import server

    row = store.sessions.get(chat.session_id)
    draft = _local_draft(chat.scope, chat.session_id)
    if draft is not None:
        with server._session_profile_runtime_scope(draft):
            if server._session_uses_compute_host(draft, server._load_dashboard_process_isolation_config()):
                return None
    if row is None:
        if draft is None or draft is not chat.draft or not server._ensure_session_db_row(draft):
            return None
        row = store.sessions.get(chat.session_id)
    # A delete or adoption may have landed while the provider answered the SDP offer.
    if row is None or not _is_legacy_row(store, row):
        return None
    validate_legacy_row(chat.scope.name, row.as_session_row(), subject=actor.subject, rule=owner_rule())
    tip = store.lineage.tip(chat.session_id)
    if tip != chat.session_id:
        row = store.sessions.get(tip)
        if row is None or not _is_legacy_row(store, row):
            return None
        validate_legacy_row(chat.scope.name, row.as_session_row(), subject=actor.subject, rule=owner_rule())
    if draft is not None:
        server._persist_branch_seed(draft)
    server._reopen_if_finalized(store, tip)
    return tip




async def start_local_capture(chat: LocalVoiceChat, actor: SessionActor, call_id: str, *,
                              api_key: str, base_url: str) -> bool:
    """Attach to the negotiated call, lending the TUI writer to the capture task."""
    from gateway.session_voice_capture import voice_capture_socket

    # A separate reference keeps the same backend alive if the TUI session closes mid-call.
    store = open_persistence(chat.scope)
    captures = None
    try:
        target = await asyncio.to_thread(_prepare_capture, store, chat, actor)
        if target is None:
            return False
        captures = VoiceCaptures(store.messages)
        return await captures.start(
            target, call_id, voice_capture_socket(call_id, api_key=api_key, base_url=base_url),
        )
    except (OSError, PersistenceError, RuntimeError, ValueError) as exc:
        log.warning('GPT-Live local capture unavailable (%s)', type(exc).__name__)
        return False
    finally:
        task = captures.tasks.get(call_id) if captures is not None else None
        if task is None or task.done():
            await asyncio.to_thread(store.close)
        else:
            loop = asyncio.get_running_loop()

            def finished(_):
                loop.run_in_executor(None, store.close)

            task.add_done_callback(finished)
