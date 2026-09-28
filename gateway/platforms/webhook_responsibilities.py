"""Responsibility ingress on the native webhook HTTP listener and turn dispatcher."""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from aiohttp import web
from responsibilities import webhook_store as store
from responsibilities.webhook_contract import build_webhook_prompt, render_stream_key, resolve_handshake, verify_delivery_signature
from responsibilities.secrets import resolve_secret_reference
from responsibilities.common import get_responsibilities_root
from responsibilities.packages import read_responsibility_package

logger = logging.getLogger(__name__)


class ResponsibilityIngress:
    def __init__(self, adapter):
        self.adapter = adapter
        self.active = {}
        self.initialized = set()
        from hermes_constants import get_hermes_home
        self.home = get_hermes_home()

    async def handle(self, request):
        from gateway.run import _profile_runtime_scope
        from gateway.platforms.webhook import _PROFILE_REJECTED
        from hermes_cli.profiles import get_profile_dir
        with _profile_runtime_scope(self.home):
            profile = self.adapter._resolve_request_profile(request) if request.match_info.get('profile') else None
        if profile is _PROFILE_REJECTED:
            raise web.HTTPNotFound()
        home = get_profile_dir(profile) if profile else self.home
        with _profile_runtime_scope(home):
            if home != self.home and not self._enabled():
                raise web.HTTPNotFound()
            return await self._handle(request)

    async def _handle(self, request):
        token = request.match_info['token']
        route = await asyncio.to_thread(store.route, token)
        if not route:
            raise web.HTTPNotFound()
        raw = await request.read()
        if len(raw) > self.adapter._max_body_bytes:
            raise web.HTTPRequestEntityTooLarge(max_size=self.adapter._max_body_bytes, actual_size=len(raw))
        try:
            payload = json.loads(raw) if raw else {}
        except (UnicodeError, ValueError):
            payload = raw.decode('utf-8', errors='replace')
        definition = json.loads(route['definition'])
        try:
            handshake = resolve_handshake(definition.get('handshake'), method=request.method,
                query=request.query, body=payload, resolve_secret=resolve_secret_reference)
            if handshake:
                mime, body = handshake
                return web.json_response(body) if mime == 'application/json' else web.Response(text=str(body), content_type=mime)
            if request.method != 'POST':
                raise web.HTTPMethodNotAllowed(request.method, ['POST'])
            error = verify_delivery_signature(definition['verify'], raw=raw, headers=request.headers, resolve_secret=resolve_secret_reference) if definition.get('verify') else None
        except RuntimeError:
            raise web.HTTPServiceUnavailable(text='Webhook secret unavailable')
        if error:
            raise web.HTTPForbidden(text='signature_invalid')
        stream, warning = render_stream_key(payload, definition.get('key'))
        dedup = next((request.headers[name] for name in ('X-GitHub-Delivery','X-Webhook-Delivery','I-Twilio-Idempotency-Token','X-Request-ID','ce-id') if request.headers.get(name)), None)
        if not dedup and isinstance(payload, dict):
            # Event envelopes carry event IDs; an arbitrary resource "id" is not a retry ID.
            if payload.get('type') == 'event_callback' and isinstance(payload.get('event_id'), str):
                dedup = payload['event_id']
            elif payload.get('object') == 'event' and isinstance(payload.get('id'), str) and payload['id'].startswith('evt_'):
                dedup = payload['id']
        dedup = dedup or uuid.uuid4().hex
        delivery = dict(payload=payload, provider_delivery_id=dedup, received_at=datetime.now(timezone.utc).isoformat(),
                        content_type=request.content_type, event_type=request.headers.get('X-GitHub-Event',''))
        status = await asyncio.to_thread(store.accept, token, stream, dedup, delivery)
        if status == 'rate_limited':
            return web.json_response({'status':status}, status=429)
        ack = definition.get('ack')
        if ack:
            return web.Response(status=ack.get('status',200), text=ack.get('body',''), content_type=ack.get('content_type','text/plain'))
        return web.json_response({'accepted': True, **({'warning':warning} if warning else {})}, status=202)

    @staticmethod
    def _enabled():
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config().platforms.get(Platform.WEBHOOK)
        return bool(config and config.enabled)

    def _profiles(self):
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import profiles_to_serve
        config = getattr(getattr(self.adapter, 'gateway_runner', None), 'config', None)
        if not getattr(config, 'multiplex_profiles', False):
            return [(None, self.home)]
        with _profile_runtime_scope(self.home):
            return profiles_to_serve(multiplex=True)

    async def run(self):
        from gateway.run import _profile_runtime_scope
        while True:
            for profile, home in self._profiles():
                try:
                    with _profile_runtime_scope(home):
                        if home == self.home or self._enabled():
                            await asyncio.to_thread(store.reconcile)
                            self._dispatch_profile(profile, home)
                except Exception:
                    logger.exception('Responsibility webhook dispatch failed for %s', home)
            await asyncio.sleep(1)

    def _dispatch_profile(self, profile, home):
        if home not in self.initialized:
            # A process crash has unknown side effects. Never replay claimed work blindly.
            with store.database() as db:
                db.execute("UPDATE deliveries SET state='interrupted' WHERE state='running'")
            self.initialized.add(home)
        for token, stream in store.streams():
            key = hashlib.sha256((str(home)+'\0'+token+'\0'+stream).encode()).hexdigest()[:24]
            if key in self.active:
                continue
            row = store.route(token)
            if not row:
                continue
            package = read_responsibility_package(get_responsibilities_root(), row['package'])
            if not package.get('found') or package.get('malformed'):
                continue
            ids, deliveries = store.claim(token,stream)
            if not ids:
                continue
            self.active[key] = (home, ids)
            try:
                definition = json.loads(row['definition'])
                report = definition.get('deliver','muted')
                route = dict(responsibility=str(get_responsibilities_root()/row['package']), declaration=row['name'],
                             definition=definition, deliver='local' if report=='muted' else report)
                prompt = build_webhook_prompt(route=route, responsibility_document=package['responsibility_document'],
                    state_document=package['state_document'], deliveries=deliveries, state_nonempty=bool(package['state_entries']))
                platform, _, target = report.partition(':')
                chat, sep, thread = target.partition(':')
                config = {'deliver': 'log' if report in ('muted', 'local') else platform,
                          'deliver_extra': {'chat_id':chat, **({'thread_id':thread} if sep else {})},
                          'toolsets':['hermes-cli'], 'responsibility_stream':key, 'mirror_to_session': True}
                self.adapter._spawn_agent_run(deliveries, prompt, str(ids[-1]), time.time(), route_config=config,
                        route_name='responsibility-'+key, profile=profile, event_type='responsibility')
            except Exception:
                store.complete(self.active.pop(key)[1], state='failed')
                raise
    def completed(self, source, outcome):
        from gateway.platforms.event import ProcessingOutcome
        key = str(source.chat_id).rsplit(':',1)[-1]
        if key in self.active:
            state = {ProcessingOutcome.SUCCESS: 'done', ProcessingOutcome.FAILURE: 'failed',
                     ProcessingOutcome.CANCELLED: 'interrupted'}[outcome]
            from gateway.run import _profile_runtime_scope
            home, ids = self.active.pop(key)
            with _profile_runtime_scope(home):
                store.complete(ids, state=state)
