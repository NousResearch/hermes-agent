"""Tests for the read-only Microsoft Graph mail CLI."""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path

pytest = importlib.import_module("pytest")


SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/microsoft-graph-mail/scripts"
)
GRAPH_MAIL_PATH = SCRIPT_DIR / "graph_mail.py"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None, headers=None):
        self.requests.append({"url": url, "params": params, "headers": headers})
        return self.responses.pop(0)


class FakeClientFactory:
    def __init__(self, responses):
        self.client = FakeClient(responses)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.client


@pytest.fixture
def graph_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("graph_mail_test", GRAPH_MAIL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_valid_access_token", lambda: "secret-token")
    return module


def test_search_messages_uses_me_endpoint_and_returns_summary(graph_module, monkeypatch, capsys):
    factory = FakeClientFactory(
        [
            FakeResponse(
                payload={
                    "value": [
                        {
                            "id": "message-id",
                            "conversationId": "conversation-id",
                            "from": {"emailAddress": {"name": "Alice", "address": "alice@example.com"}},
                            "toRecipients": [{"emailAddress": {"address": "bob@example.com"}}],
                            "subject": "Planning",
                            "receivedDateTime": "2026-05-14T12:00:00Z",
                            "bodyPreview": "Let's plan",
                            "isRead": False,
                            "importance": "normal",
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setattr(graph_module.httpx, "Client", factory)

    graph_module.search_messages(argparse.Namespace(query="planning", max=10))

    request = factory.client.requests[0]
    assert request["url"] == "https://graph.microsoft.com/v1.0/me/messages"
    assert "/users/" not in request["url"]
    assert request["params"]["$search"] == '"planning"'
    assert request["headers"]["Authorization"] == "Bearer secret-token"

    output = json.loads(capsys.readouterr().out)
    assert output == [
        {
            "id": "message-id",
            "conversationId": "conversation-id",
            "from": "Alice <alice@example.com>",
            "to": ["bob@example.com"],
            "subject": "Planning",
            "date": "2026-05-14T12:00:00Z",
            "snippet": "Let's plan",
            "isRead": False,
            "importance": "normal",
        }
    ]


def test_list_messages_follows_next_link_until_limit(graph_module, monkeypatch, capsys):
    factory = FakeClientFactory(
        [
            FakeResponse(payload={"value": [{"id": "1"}], "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/messages?page=2"}),
            FakeResponse(payload={"value": [{"id": "2"}]}),
        ]
    )
    monkeypatch.setattr(graph_module.httpx, "Client", factory)

    graph_module.list_messages(argparse.Namespace(max=2))

    assert factory.client.requests[0]["url"] == "https://graph.microsoft.com/v1.0/me/messages"
    assert factory.client.requests[1]["url"] == "https://graph.microsoft.com/v1.0/me/messages?page=2"
    output = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in output] == ["1", "2"]


def test_list_messages_rejects_next_link_outside_graph_origin(graph_module, monkeypatch, capsys):
    factory = FakeClientFactory(
        [
            FakeResponse(payload={"value": [{"id": "1"}], "@odata.nextLink": "https://example.com/me/messages?page=2"}),
        ]
    )
    monkeypatch.setattr(graph_module.httpx, "Client", factory)

    with pytest.raises(SystemExit):
        graph_module.list_messages(argparse.Namespace(max=2))

    captured = capsys.readouterr()
    assert "outside https://graph.microsoft.com" in captured.err
    assert len(factory.client.requests) == 1


def test_get_message_strips_html_and_url_encodes_id(graph_module, monkeypatch, capsys):
    factory = FakeClientFactory(
        [
            FakeResponse(
                payload={
                    "id": "A/B",
                    "conversationId": "thread",
                    "from": {"emailAddress": {"address": "alice@example.com"}},
                    "toRecipients": [],
                    "ccRecipients": [],
                    "subject": "HTML",
                    "receivedDateTime": "2026-05-14T12:00:00Z",
                    "bodyPreview": "Hello",
                    "body": {"contentType": "html", "content": "<p>Hello&nbsp;<b>world</b></p>"},
                    "webLink": "https://outlook.office.com/mail/item",
                }
            )
        ]
    )
    monkeypatch.setattr(graph_module.httpx, "Client", factory)

    graph_module.get_message(argparse.Namespace(message_id="A/B"))

    assert factory.client.requests[0]["url"].endswith("/me/messages/A%2FB")
    output = json.loads(capsys.readouterr().out)
    assert output["body"] == "Hello world"
    assert output["webLink"] == "https://outlook.office.com/mail/item"


def test_graph_errors_do_not_print_access_token(graph_module, monkeypatch, capsys):
    factory = FakeClientFactory(
        [FakeResponse(status_code=403, payload={"error": {"message": "Denied"}})]
    )
    monkeypatch.setattr(graph_module.httpx, "Client", factory)

    with pytest.raises(SystemExit):
        graph_module.list_messages(argparse.Namespace(max=1))

    err = capsys.readouterr().err
    assert "Denied" in err
    assert "secret-token" not in err


def test_limits_requested_result_count(graph_module, monkeypatch):
    factory = FakeClientFactory([FakeResponse(payload={"value": []})])
    monkeypatch.setattr(graph_module.httpx, "Client", factory)

    graph_module.list_messages(argparse.Namespace(max=500))

    assert factory.client.requests[0]["params"]["$top"] == 25