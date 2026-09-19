"""A recording HTTP transport and its canned response, shared by the connector client suites."""

from __future__ import annotations


class FakeResponse:
    def __init__(self, status_code, body=None, *, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeTransport:
    """Records requests; replays queued responses (exceptions raise)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.requests.append(
            {"method": method, "url": url, "headers": dict(headers or {}), "json": json, "timeout": timeout}
        )
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
