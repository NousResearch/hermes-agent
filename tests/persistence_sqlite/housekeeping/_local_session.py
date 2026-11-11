"""A committed local session and a receipted runtime mutation on a SQLite backend store's holder, for the housekeeping tests of local reset lineages."""
from dataclasses import asdict


def _local_session(db, epoch, cwd):
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_policy import build_policy, created_surface
    from persistence.authority.bindings import local_session_id
    from persistence.sqlite.authority._local import commit_local_session
    sid = local_session_id('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(cwd), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)},
        client_request=None, created_source=created_surface(policy.source))
    return sid


def _mutate(db, epoch, sid, operation, **extra):
    from persistence.sqlite.authority._mutation import mutate_runtime_session
    from persistence.sqlite.conversation._store.sessions import get_session
    session = get_session(db, sid)
    return mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=sid,
        request_id=operation + '-' + str(session['runtime_revision']), operation=operation,
        expected_revision=session['runtime_revision'], expected_generation=session['runtime_generation'],
        payload={}, **extra)[0]
