"""Control protocol scheduling contract, independent of the native pipe transport."""

import asyncio
import json

from gateway.control_socket import GatewayControlServer, _PipeControlProtocol


def test_pipe_dispatch_keeps_owning_loop_available_and_answers_once(tmp_path):
    async def scenario():
        loop = asyncio.get_running_loop()
        admitted = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()
        responses = []

        async def quiesce():
            admitted.set()
            await release.wait()
            return {"quiesced": True}

        def handler():
            return asyncio.run_coroutine_threadsafe(quiesce(), loop).result(timeout=2.0)

        class Transport:
            def write(self, data):
                responses.append(data)

            def close(self):
                closed.set()

        protocol = _PipeControlProtocol(GatewayControlServer(
            tmp_path, verb_handlers={"quiesce": handler}))
        protocol.connection_made(Transport())
        protocol.data_received(b'{"verb":"qui')
        protocol.data_received(b'esce","id":1}\n')
        try:
            await asyncio.wait_for(admitted.wait(), timeout=2.0)
            assert responses == []
            protocol.data_received(b'{"verb":"quiesce","id":2}\n')
        finally:
            release.set()
        await asyncio.wait_for(closed.wait(), timeout=2.0)
        assert len(responses) == 1
        answer = json.loads(responses[0])
        assert answer["id"] == 1
        assert answer["result"] == {"quiesced": True}

    asyncio.run(scenario())
