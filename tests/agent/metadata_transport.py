"""Socket-free response data for tests that exercise the real metadata transport."""
import httpx
import pytest


@pytest.fixture
def metadata_transport(monkeypatch):
    from agent import model_metadata

    responses, requests = [], []
    real_client = httpx.Client

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        **kw, transport=httpx.MockTransport(handle),
    ))
    monkeypatch.setattr(model_metadata, "detect_local_server_type", lambda *a, **kw: None)
    return responses, requests
