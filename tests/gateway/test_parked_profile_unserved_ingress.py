"""Ingress the parked-profile checks miss when they read only the URL prefix and the parked map.

A coalesced webhook is acknowledged before admission applies ``profile_routes``; a ``/p/<profile>/``
prefix can name a profile on disk that this gateway does not serve for another reason; and the API
doors and the shared listener answer a parked profile's prefix with a final 409 or 404. Each is a
retryable 503 here, so the client's retry is processed once the profile is served again.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.gateway.test_parked_profile_ingress import (
    _admissions, _cancel, _close, _homes, _runner_with_beta_parked, _serve_beta_again, _webhook)
from tests.waiting import async_wait_until


def _request(delivery_id, body, profile):
    raw = json.dumps(body).encode()
    request = MagicMock()
    request.headers = {'X-GitHub-Delivery': delivery_id}
    request.content_length = len(raw)
    request.match_info = {'route_name': 'events', 'profile': profile}
    request.method = 'POST'

    async def read():
        return raw

    request.read = read
    return request


@pytest.mark.asyncio
async def test_a_coalesced_webhook_profile_routes_send_to_a_parked_profile_is_retried_not_lost(
        tmp_path, monkeypatch):
    """A bare ``/webhooks/events`` that ``profile_routes`` sends to beta: a 202 would never be retried
    by the provider, and the flush is refused while beta is parked."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    from gateway.profile_routing import ProfileRoute
    homes = _homes(tmp_path, monkeypatch)
    beta = homes[1][1].resolve()
    routes = [ProfileRoute(name='hooks', platform='webhook', profile='beta', user_id='webhook:events')]
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes, profile_routes=routes)
    adapter = _webhook(runner, {'events': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}',
                                           'coalesce': {'key': 'id', 'window_seconds': 0.1}}})
    settled = lambda: not adapter._coalescer.pending and not adapter._background_tasks
    try:
        first = await adapter._handle_webhook(_request('delivery-1', {'id': 17}, ''))
        assert first.status == 503 and first.headers.get('Retry-After'), (first.status, first.text)
        assert 'beta' not in first.text, 'the refusal names the routed profile'
        assert adapter._seen_deliveries == {}, 'the refused delivery id was recorded'
        await _serve_beta_again(runner, beta)
        retry = await adapter._handle_webhook(_request('delivery-1', {'id': 17}, ''))
        await async_wait_until(settled, what='the retried batch dispatched')
        assert json.loads(retry.text)['status'] == 'coalesced', retry.text
        assert _admissions(beta) == 1, 'delivery-1 was never admitted'
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coalesce', 'deliver_only', 'cron_job'])
async def test_a_webhook_prefix_with_no_serving_authority_is_retried(tmp_path, monkeypatch, mode):
    """``/p/beta/`` while beta is on disk but not served and not parked (its own gateway still runs,
    or the watcher has not picked it up): the live registry, not the parked map, decides."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    from gateway.run_runtime import unpark_profile
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes)
    unpark_profile(runner, 'beta')
    route = {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'profile': 'beta',
             mode: {'key': 'id', 'window_seconds': 0.1} if mode == 'coalesce' else True}
    if mode == 'deliver_only':
        route['deliver'] = 'log'
    adapter = _webhook(runner, {'events': route})
    try:
        refused = await adapter._handle_webhook(_request('delivery-1', {'id': 17}, 'beta'))
        assert refused.status == 503 and refused.headers.get('Retry-After'), (refused.status, refused.text)
        assert adapter._seen_deliveries == {}, 'the refused delivery id was recorded'
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
async def test_api_and_shared_listener_requests_to_a_parked_profile_are_retryable(tmp_path, monkeypatch):
    """``/p/beta/v1/...`` (chat completions and runs, Responses' admission lookup) and
    ``/p/beta/<platform callback>`` answer 503 while beta is parked, not 409 or 404."""
    from aiohttp import web
    from gateway.platforms.api_server_authority_runs import store_error_response
    from gateway.platforms.api_server_response_admissions import find_admission
    from gateway.platforms.shared_ingress import dispatch_profile_ingress
    from gateway.run import _profile_runtime_scope
    from gateway.session_api_turn import admit_api_steps, admit_api_turn_async
    from hermes_state_runtime import RuntimeStoreError
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes)
    api = SimpleNamespace(gateway_runner=runner)
    refusals = []
    try:
        with _profile_runtime_scope(homes[1][1].resolve(), hydrate_secrets=False):
            for door in (lambda: admit_api_turn_async(api), lambda: next(admit_api_steps(api)),
                         lambda: find_admission(api, ('key',), 'request')):
                with pytest.raises(RuntimeStoreError) as refused:
                    pending = door()
                    if hasattr(pending, '__await__'):
                        await pending
                refusals.append(refused.value)
        assert [exc.reason for exc in refusals] == ['runtime_draining'] * 3
        answer = store_error_response(lambda message, **kw: {'error': {'message': message, **kw}}, refusals[0])
        assert answer.status == 503 and answer.headers.get('Retry-After')
        with pytest.raises(web.HTTPServiceUnavailable):
            await dispatch_profile_ingress(runner, 'beta', 'line/webhook', MagicMock())
    finally:
        _close(runner, homes)
