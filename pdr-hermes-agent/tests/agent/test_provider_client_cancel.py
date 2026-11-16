"""Request-owned non-HTTP transports must participate in interruption."""

import subprocess
import sys
import threading

from agent.client_lifecycle import ClientLifecycleMixin


class Agent(ClientLifecycleMixin):
    socket_sweeps = 0

    def _client_log_context(self):
        return "provider=test"

    def _force_close_tcp_sockets(self, client):
        self.socket_sweeps += 1
        return 0


def test_cross_thread_cancel_stops_real_provider_process():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    class Client:
        def cancel(self):
            proc.terminate()

    agent = Agent()
    client = Client()
    try:
        thread = threading.Thread(
            target=agent._abort_request_openai_client,
            args=(client,),
            kwargs={"reason": "interrupt_abort"},
        )
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert proc.wait(timeout=2) is not None
        assert agent.socket_sweeps == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_dynamic_attributes_do_not_opt_http_clients_into_cancel():
    calls = []

    class Client:
        def __getattr__(self, name):
            return lambda: calls.append(name)

    agent = Agent()
    agent._abort_request_openai_client(Client(), reason="interrupt_abort")
    assert agent.socket_sweeps == 1
    assert calls == []
