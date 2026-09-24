"""Host-owned, exact-grant read projections for Telegram-authenticated task views."""
from __future__ import annotations

import asyncio
import base64
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import unicodedata

from gateway.telegram_init_data import InitDataDenied, InitDataExpired, verify_init_data


class TaskReadDenied(ValueError):
    def __init__(self, reason='unavailable'):
        self.reason = reason if reason in {'expired', 'temporary'} else 'unavailable'
        super().__init__(self.reason)


def decode_selector(value):
    try:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,512}', value):
            raise ValueError
        raw = base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)
        if base64.urlsafe_b64encode(raw).decode().rstrip('=') != value:
            raise ValueError
        fields = json.loads(raw)
        if (not isinstance(fields, list) or len(fields) != 4
                or any(not isinstance(v, str) for v in fields[:3])
                or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', fields[0])
                or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', fields[1])
                or not re.fullmatch(r'(?:t_[a-f0-9]{8,64}|p_[a-f0-9]{16,64})', fields[2])
                or type(fields[3]) is not int or not 0 < fields[3] < 2**63):
            raise ValueError
        return tuple(fields)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise TaskReadDenied('unavailable') from None


def _clean_title(value):
    text = str(value or '')[:4096]
    text = re.sub(r'(?i)(?:https?://\S+|(?:token|password|secret|api[_-]?key)\s*[:=]\s*\S+|(?<!\w)(?:sk-|ghp_|github_pat_)[\w-]+|\b\d{6,}:[\w-]{20,})', '[redacted]', text)
    text = re.sub(r'(?<!\w)(?:~?/[^\s]+|[A-Za-z]:[\\/][^\s]+)', '[local path]', text)
    text = ' '.join(''.join(c if not unicodedata.category(c).startswith('C') else ' ' for c in text).split())
    return text[:240] or 'Untitled task'


class TaskReadService:
    """No action rights. Configuration and native owner are rechecked at disclosure.

    GET never changes canonical task/event/action/receipt/dependency data or creates
    config backups. SQLite mode=ro may create bounded WAL/shm coordination files;
    it must see committed WAL data, so immutable=1 is not appropriate here.
    """
    def __init__(self, ctx, scope, proposal_provider=None):
        from hermes_constants import get_hermes_home
        from hermes_cli.profiles import profile_matches_home
        self.home = Path(get_hermes_home()).resolve()
        profiles = {route.profile for route in scope.routes}
        if len(profiles) != 1:
            raise ValueError("task detail scope must name exactly one runtime profile")
        self.profile = next(iter(profiles))
        if not profile_matches_home(self.profile, self.home):
            raise ValueError("task detail scope profile must match its owning API-server profile")
        self.scope = scope
        self.plugin_id = ctx.plugin_id
        self.active = True
        self.lock = threading.RLock()
        self.binding = None
        self.proposal_provider = proposal_provider

    def close(self):
        with self.lock:
            self.active = False
            self.binding = None

    def bind(self, app, adapter):
        from hermes_constants import get_hermes_home
        with self.lock:
            if not self.active or Path(get_hermes_home()).resolve() != self.home:
                raise TaskReadDenied('Task detail requires its owning API-server profile')
            if '*' in adapter._cors_origins:
                raise TaskReadDenied('Task detail requires explicit API-server CORS origins, not wildcard')
            self.binding = (app, adapter)

    def _policy(self):
        from hermes_constants import get_hermes_home
        from hermes_cli.config_effective import load_user_config_effective
        from yaml import YAMLError
        if (not self.active or self.binding is None
                or Path(get_hermes_home()).resolve() != self.home or not self.home.is_dir()):
            raise TaskReadDenied('unavailable')
        try:
            # Authorization must never recover an obsolete grant from last-good config.
            # Authorization reads must not create active-home config backups.
            cfg = load_user_config_effective(fail_closed=True, side_effect_free=True) or {}
        except YAMLError:
            raise TaskReadDenied('unavailable') from None
        plugins = cfg.get('plugins', {})
        settings = plugins.get('entries', {}).get(self.plugin_id, {}).get('settings', {})
        if (self.plugin_id not in plugins.get('enabled', [])
                or self.plugin_id in plugins.get('disabled', [])
                or settings.get('enabled') is not True or settings.get('task_detail') is not True):
            raise TaskReadDenied('unavailable')
        if self.proposal_provider is not None:
            from gateway.surface_scope import parse_surface_scope
            if settings.get('work_briefs') is not True:
                raise TaskReadDenied('unavailable')
            try:
                parse_surface_scope(settings.get('scope'), require_tasks=False)
            except (TypeError, ValueError):
                raise TaskReadDenied('unavailable') from None
        kanban = cfg.get('kanban', {})
        grants = kanban.get('read_grants', [])
        if not isinstance(grants, list) or len(grants) > 4096:
            raise TaskReadDenied('unavailable')
        return settings, grants, kanban.get('read_max_age_seconds', 300)

    def available(self, app):
        try:
            with self.lock:
                self._policy()
                return self.binding is not None and self.binding[0] is app
        except (TaskReadDenied, AttributeError, TypeError, ValueError, OSError):
            return False

    def _authorize(self, app, raw, selector):
        from gateway.config import Platform
        settings, grants, age = self._policy()
        if self.binding is None or self.binding[0] is not app:
            raise TaskReadDenied('unavailable')
        api = self.binding[1]
        runner = api.gateway_runner
        if runner is None:
            raise TaskReadDenied('unavailable')
        owner = runner._authorization_adapter(Platform.TELEGRAM, self.profile)
        # Delivery resolution intentionally permits shared-bot satellite fallback.
        # Read authentication requires this profile's own credential instead.
        if self.profile != getattr(runner, '_primary_profile_name', None):
            native_owner = getattr(runner, '_profile_adapters', {}).get(self.profile, {}).get(Platform.TELEGRAM)
            if native_owner is None or owner is not native_owner:
                raise TaskReadDenied('unavailable')
        if owner is None or not owner.config.token:
            raise TaskReadDenied('unavailable')
        principal = verify_init_data(raw, owner.config.token, now=time.time(), max_age=age)
        profile, board, tid, incarnation = selector
        if profile != self.profile:
            raise TaskReadDenied('unavailable')
        required = dict(actor=principal['actor'], bot_id=principal['bot_id'], profile=profile,
                        board=board, task_id=tid, task_incarnation=incarnation, permissions=['read'])
        candidates = [g for g in grants if isinstance(g, dict)
                      and all(g.get(k) == v and type(g.get(k)) is type(v) for k, v in required.items())]
        legacy = (self.proposal_provider is None
                  and self.scope.allows_task(profile, board, tid) and len(candidates) == 1)
        if self.proposal_provider is None and not legacy:
            raise TaskReadDenied('unavailable')
        return (json.dumps([settings, grants, age], sort_keys=True), owner,
                owner.config.token, getattr(owner, '_bot', None),
                getattr(owner, '_live_todo_epoch', None), self.binding,
                principal['actor'], principal['bot_id'], legacy)

    def _project(self, selector):
        from hermes_cli import kanban_db as kb, kanban_db_surface as source
        from gateway.work_presentation import audience_from_dict, validate_presentation, validate_steps
        import os
        profile, board, tid, incarnation = selector
        if tid.startswith('p_'):
            if self.proposal_provider is None:
                raise TaskReadDenied('unavailable')
            try:
                result, audience = self.proposal_provider.project_proposal(selector)
            except (ValueError, KeyError, TypeError, RuntimeError):
                raise TaskReadDenied('unavailable') from None
            identity = ('proposal', tid, incarnation, result['revision'], result['updated_at'])
            return result, audience, identity
        # Explicit board authority must not alias every slug to a worker's pinned DB.
        if os.environ.get('HERMES_KANBAN_DB'):
            raise TaskReadDenied('unavailable')
        path = kb.kanban_db_path(board=board).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise TaskReadDenied('unavailable')
        before = path.stat()
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0.1)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA query_only=ON')
            conn.execute('BEGIN')
            identity = source.get_task_source(conn, tid, task_incarnation=incarnation)
            # No SELECT * or task/body deserialization; only the documented projection.
            task = conn.execute('SELECT title, status FROM tasks WHERE id=?', (tid,)).fetchone()
            event = conn.execute('SELECT created_at FROM task_events WHERE id=?', (identity.current_revision,)).fetchone()
            if (task is None or task['status'] not in kb.VALID_STATUSES or event is None
                    or type(event[0]) is not int or not 0 <= event[0] < 253402300800):
                raise TaskReadDenied('unavailable')
            result = dict(title=_clean_title(task['title']), status=task['status'],
                          incarnation=identity.task_incarnation, revision=identity.current_revision,
                          updated_at=event[0])
            audience = None
            published = conn.execute(
                "SELECT id, run_id, payload, created_at FROM task_events "
                "WHERE task_id=? AND json_extract(payload, '$.publication.version')=1 "
                "ORDER BY id DESC LIMIT 1", (tid,),
            ).fetchone() if self.proposal_provider is not None else None
            if published is not None:
                try:
                    payload = json.loads(published['payload'])
                    publication = payload['publication']
                    if (type(publication) is not dict or publication.get('version') != 1
                            or publication.get('run_id') != published['run_id']):
                        raise ValueError
                    audience = audience_from_dict(publication['audience'])
                    presentation = validate_presentation(publication['presentation'])
                    steps = validate_steps(publication.get('steps', []))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    raise TaskReadDenied('unavailable') from None
                result.update(
                    kind='task',
                    presentation=presentation, steps=steps,
                    published_at=published['created_at'],
                    publication_stale=published['id'] != identity.current_revision,
                )
        after = path.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise TaskReadDenied('unavailable')
        return result, audience, (str(path), after.st_dev, after.st_ino,
                                  identity.task_incarnation, identity.current_revision)

    def _audience(self, selector):
        """Read only exact authorization metadata before any content projection."""
        from hermes_cli import kanban_db as kb, kanban_db_surface as source
        from hermes_cli.kanban_publication import latest_task_audience
        import os
        profile, board, tid, incarnation = selector
        if tid.startswith('p_'):
            if self.proposal_provider is None:
                raise TaskReadDenied('unavailable')
            try:
                return self.proposal_provider.proposal_audience(selector)
            except (ValueError, KeyError, TypeError, RuntimeError):
                raise TaskReadDenied('unavailable') from None
        if os.environ.get('HERMES_KANBAN_DB'):
            raise TaskReadDenied('unavailable')
        path = kb.kanban_db_path(board=board).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise TaskReadDenied('unavailable')
        before = path.stat()
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0.1)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA query_only=ON')
            identity = source.get_task_source(conn, tid, task_incarnation=incarnation)
            audience = latest_task_audience(conn, tid)
        after = path.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise TaskReadDenied('unavailable')
        return audience, (str(path), after.st_dev, after.st_ino,
                          identity.task_incarnation, identity.current_revision)

    async def _member(self, bot, actor, bot_id, audience):
        current_scope = self.scope
        if self.proposal_provider is not None:
            from gateway.surface_scope import parse_surface_scope
            settings, _, _ = self._policy()
            try:
                current_scope = parse_surface_scope(settings.get('scope'), require_tasks=False)
            except (TypeError, ValueError):
                raise TaskReadDenied('unavailable') from None
        if (bot is None or audience.bot_id != bot_id or audience.profile != self.profile
                or not self.scope.allows_route(audience.profile, audience.platform,
                                               audience.chat_id, audience.thread_id)
                or not current_scope.allows_route(audience.profile, audience.platform,
                                                  audience.chat_id, audience.thread_id)):
            raise TaskReadDenied('unavailable')
        try:
            bot_member = await bot.get_chat_member(audience.chat_id, bot_id)
            actor_member = await bot.get_chat_member(audience.chat_id, actor)
        except Exception:
            raise TaskReadDenied('temporary') from None
        def status(member):
            return str(getattr(member, 'status', '')).lower()
        if status(bot_member) not in {'administrator', 'creator', 'owner'}:
            raise TaskReadDenied('unavailable')
        actor_status = status(actor_member)
        if not (actor_status in {'member', 'administrator', 'creator', 'owner'}
                or (actor_status == 'restricted' and getattr(actor_member, 'is_member', False) is True)):
            raise TaskReadDenied('unavailable')

    async def detail(self, app, raw, encoded):
        """Await off-loop IO, then reacquire all current authority before disclosure.

        The second read cannot reuse the earlier SQLite snapshot: deletion/recreation
        during the awaited read must not disclose a former resource.
        """
        try:
            selector = decode_selector(encoded)
            with self.lock:
                admission = self._authorize(app, raw, selector)
            if not admission[-1]:
                audience, path_identity = await asyncio.to_thread(self._audience, selector)
                if audience is None:
                    raise TaskReadDenied('unavailable')
                await self._member(admission[3], admission[-3], admission[-2], audience)
            projected, projected_audience, projected_identity = await asyncio.to_thread(self._project, selector)
            with self.lock:
                if self._authorize(app, raw, selector) != admission:
                    raise TaskReadDenied('unavailable')
            if admission[-1]:
                current, current_audience, current_identity = self._project(selector)
                if (current_identity != projected_identity
                        or current_audience != projected_audience):
                    raise TaskReadDenied('unavailable')
            else:
                if projected_identity != path_identity or projected_audience != audience:
                    raise TaskReadDenied('unavailable')
                current, current_audience, current_identity = self._project(selector)
                if current_identity != path_identity or current_audience != audience:
                    raise TaskReadDenied('unavailable')
            if not admission[-1]:
                await self._member(admission[3], admission[-3], admission[-2], audience)
                with self.lock:
                    if self._authorize(app, raw, selector) != admission:
                        raise TaskReadDenied('unavailable')
                final_audience, final_identity = await asyncio.to_thread(self._audience, selector)
                if final_identity != path_identity or final_audience != audience:
                    raise TaskReadDenied('unavailable')
            return dict(current, selector=encoded, fetched_at=int(time.time()))
        except InitDataExpired:
            raise TaskReadDenied('expired') from None
        except TaskReadDenied as exc:
            raise TaskReadDenied(exc.reason) from None
        except (InitDataDenied, sqlite3.Error, OSError, ValueError, TypeError, RuntimeError,
                AttributeError, KeyError):
            raise TaskReadDenied('unavailable') from None


def register_task_detail(ctx, *, scope):
    from gateway.surface_scope import parse_surface_scope
    parsed_scope = parse_surface_scope(scope, require_tasks=True)
    current = getattr(ctx._manager, '_task_read_registration', None)
    if current is not None and current.active:
        if current.plugin_id != ctx.plugin_id:
            raise ValueError('A task detail consumer already owns this profile')
        if current.scope != parsed_scope:
            raise ValueError('Task detail scope is already bound for this profile')
        return current
    service = TaskReadService(ctx, parsed_scope)
    ctx._manager._task_read_registration = service
    ctx.on_unload(service.close)
    return service
