"""Invariant checks against real httpx, SDK and the selected Hermes source overlay."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from agent import auxiliary_hooks as aux
from agent import process_bootstrap as bootstrap

RAW = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
       'prompt_tokens_details': {'cached_tokens': 40},
       'completion_tokens_details': {'reasoning_tokens': 7}}


class ObserverTests(unittest.TestCase):
    def test_terminal_events_keep_the_owning_context_after_stream_handoff(self):
        import contextvars
        from agent import usage_observer as observer

        owner = contextvars.ContextVar('observer-test-owner', default='unset')
        records = []
        def fire(event, **payload):
            records.append((event, owner.get()))

        first = owner.set('profile-a')
        try:
            with patch.object(aux, '_has_hook', return_value=True), patch.object(aux, '_fire', side_effect=fire):
                stream = aux.run_with_aux_hooks(lambda: iter([SimpleNamespace(usage=None)]),
                    aux_task='fixture', metadata={}, client=None, kwargs={},
                    provider='fixture', model='fixture', api_mode='chat_completions', streaming=True)
                second = owner.set('profile-b')
                try:
                    list(stream)
                finally:
                    owner.reset(second)
            self.assertIn(('auxiliary_stream_end', 'profile-a'), records)
            records.clear()
            with patch.object(observer, '_enabled', return_value=True), patch.object(observer, '_fire', side_effect=fire):
                class Body(httpx.SyncByteStream):
                    def __iter__(self):
                        yield b'fixture'
                with httpx.Client(transport=observer.SyncTransport(httpx.MockTransport(
                        lambda request: httpx.Response(200, stream=Body())))) as client:
                    response = client.send(client.build_request('GET', 'https://fixture.invalid'), stream=True)
                    second = owner.set('profile-b')
                    try:
                        self.assertEqual(response.read(), b'fixture')
                        response.close()
                    finally:
                        owner.reset(second)
            self.assertIn(('http_attempt_end', 'profile-a'), records)
        finally:
            owner.reset(first)

    def test_aux_error_keeps_status(self):
        events = []
        exc = httpx.HTTPStatusError('private-error', request=httpx.Request('POST', 'https://fixture.invalid'),
                                    response=httpx.Response(401))
        exc.status_code = 401
        with patch.object(aux, '_has_hook', return_value=True), patch.object(aux, '_fire', side_effect=lambda e, **p: events.append((e, p))):
            with self.assertRaises(httpx.HTTPStatusError) as raised:
                aux.run_with_aux_hooks(lambda: (_ for _ in ()).throw(exc), aux_task='fixture',
                    metadata={'api_request_id': 'error'}, client=None, kwargs={},
                    provider='fixture', model='fixture', api_mode='chat_completions')
        self.assertIs(raised.exception, exc)
        self.assertEqual(events[-1][1].get('status_code'), 401)
        self.assertIsNone(events[-1][1]['usage'])

    def test_stream_is_lazy_and_final_usage_is_not_handoff(self):
        events, consumed = [], []
        def source():
            consumed.append(1)
            yield SimpleNamespace(id='fixture-response', usage=None)
            consumed.append(2)
            yield SimpleNamespace(id='fixture-response', usage=SimpleNamespace(**RAW))
        with patch.object(aux, '_has_hook', return_value=True), patch.object(aux, '_fire', side_effect=lambda e, **p: events.append((e, p))):
            stream = aux.run_with_aux_hooks(source, aux_task='fixture', metadata={'api_request_id': 'stream'},
                client=None, kwargs={}, provider='fixture', model='fixture', api_mode='chat_completions', streaming=True)
            self.assertEqual(consumed, [])
            self.assertFalse(any(e == 'auxiliary_stream_end' for e, _ in events))
            chunks = list(stream)
        final = [p for e, p in events if e == 'auxiliary_stream_end']
        self.assertEqual(len(final), 1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(final[0]['raw_provider_usage'], RAW)
        self.assertEqual(final[0]['usage']['input_tokens'], 60)
        self.assertEqual(final[0]['usage']['total_tokens'], 120)
        self.assertEqual(final[0]['provider_response_id'], 'fixture-response')

    def test_client_factory_installs_observer_without_changing_shared_pool(self):
        from agent import usage_observer as observer
        events = []
        with patch.object(observer, '_enabled', return_value=True), patch.object(observer, '_fire', side_effect=lambda e, **p: events.append((e, p))):
            with patch.object(bootstrap, '_get_proxy_for_base_url', return_value=None):
                one = bootstrap.build_keepalive_http_client('http://127.0.0.1')
                two = bootstrap.build_keepalive_http_client('http://127.0.0.1')
            try:
                self.assertIsInstance(one._transport, observer.SyncTransport)
                self.assertIs(one._transport._pool, two._transport._pool)
                one.close()
                self.assertFalse(two.is_closed)
            finally:
                two.close()
                bootstrap.close_shared_transports()

    def test_native_cache_write_and_private_usage_fields(self):
        events = []
        raw = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
               'input_tokens_details': {'cached_tokens': 40, 'cache_write_tokens': 10, 'secret': 'PRIVATE'},
               'output_tokens_details': {'reasoning_tokens': 7}, 'private': 'PRIVATE'}
        source = iter([SimpleNamespace(type='response.completed',
                                      response=SimpleNamespace(id='resp-native', usage=raw))])
        with patch.object(aux, '_has_hook', return_value=True), patch.object(aux, '_fire', side_effect=lambda e, **p: events.append((e, p))):
            stream = aux.run_with_aux_hooks(lambda: source, aux_task='fixture', metadata={'api_request_id': 'native'},
                client=None, kwargs={}, provider='fixture', model='fixture', api_mode='codex_responses', streaming=True)
            list(stream)
        final = next(p for e, p in events if e == 'auxiliary_stream_end')
        self.assertEqual(final['usage']['input_tokens'], 50)
        self.assertEqual(final['usage']['cache_write_tokens'], 10)
        self.assertEqual(final['usage']['total_tokens'], 120)
        self.assertNotIn('PRIVATE', repr(final['raw_provider_usage']))

    def test_transport_reports_errors_and_preserves_exception(self):
        from agent import usage_observer as observer
        error = httpx.ConnectError('private-connect-error')
        events = []
        def fail(request):
            raise error
        with patch.object(observer, '_enabled', return_value=True), patch.object(observer, '_fire', side_effect=lambda e, **p: events.append((e, p))):
            with observer.request_scope({'api_request_id': 'connect'}):
                with httpx.Client(transport=observer.SyncTransport(httpx.MockTransport(fail))) as client:
                    with self.assertRaises(httpx.ConnectError) as raised:
                        client.get('https://fixture.invalid/private?secret=never-log')
        self.assertIs(raised.exception, error)
        self.assertEqual([e for e, _ in events], ['http_attempt_start', 'http_attempt_end'])
        self.assertEqual(events[-1][1]['error_type'], 'ConnectError')
        self.assertNotIn('private', repr(events))


    def test_async_transport_and_stream_keep_contract(self):
        from agent import usage_observer as observer
        events = []
        async def check():
            transport = observer.AsyncTransport(httpx.MockTransport(lambda r: httpx.Response(200, json={'ok': True})))
            async with httpx.AsyncClient(transport=transport) as client:
                with observer.request_scope({'api_request_id': 'async'}):
                    response = await client.get('https://fixture.invalid')
                    self.assertEqual(response.json(), {'ok': True})
            async def chunks():
                yield SimpleNamespace(id='async-response', usage=SimpleNamespace(**RAW))
            with patch.object(aux, '_has_hook', return_value=True), patch.object(aux, '_fire', side_effect=lambda e, **p: events.append((e, p))):
                stream = await aux.arun_with_aux_hooks(lambda: asyncio.sleep(0, result=chunks()),
                    aux_task='async', metadata={'api_request_id': 'async-stream'}, client=None, kwargs={'stream': True},
                    provider='fixture', model='fixture', api_mode='chat_completions')
                self.assertEqual(len([chunk async for chunk in stream]), 1)
        with patch.object(observer, '_enabled', return_value=True), patch.object(observer, '_fire', side_effect=lambda e, **p: events.append((e, p))):
            asyncio.run(check())
        self.assertEqual(len([e for e, _ in events if e == 'http_attempt_end']), 1)
        self.assertEqual(len([e for e, _ in events if e == 'auxiliary_stream_end']), 1)


if __name__ == '__main__':
    unittest.main()
