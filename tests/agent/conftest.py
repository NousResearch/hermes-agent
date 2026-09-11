"""Network isolation for compressed-argument security regression tests."""

import socket

import pytest


@pytest.fixture(autouse=True)
def _deny_integrity_test_network(request, monkeypatch):
    name = request.path.name
    if not (
        name.startswith("test_compressed_tool_arguments_")
        or name.startswith("test_tool_executor_integrity_")
        or name in {"test_tool_executor_argument_integrity.py", "test_context_compressor.py"}
    ):
        yield
        return
    # Model-catalog discovery is unrelated to the integrity contract.
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda **kwargs: {})
    attempts = []

    def denied(*args, **kwargs):
        attempts.append("outbound socket operation")
        raise AssertionError("Network forbidden in compressed-argument tests")

    for attribute in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, attribute, denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    yield
    assert not attempts, "Outbound network attempted, even if application swallowed the error"
