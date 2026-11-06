"""google_api.py turns common HttpErrors into actionable one-liners.

The bug this guards: a disabled Cloud API reached the agent as a raw Python
traceback, hiding the one fact a user needs — the activation URL.
"""

import importlib.util
import io
import json
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest

pytest.importorskip("googleapiclient")

from googleapiclient.errors import HttpError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "skills/productivity/google-workspace/scripts/google_api.py"


@pytest.fixture(scope="module")
def api():
    spec = importlib.util.spec_from_file_location("gws_api_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Resp:
    def __init__(self, status):
        self.status = status
        self.reason = ""


def _http_error(status, body):
    return HttpError(_Resp(status), json.dumps(body).encode())


def _disabled_api_error():
    return _http_error(403, {"error": {
        "message": ("Google Calendar API has not been used in project 428783091847 before "
                    "or it is disabled. Enable it by visiting https://console.developers.google.com"
                    "/apis/api/calendar-json.googleapis.com/overview?project=428783091847 then retry."),
        "errors": [{"reason": "accessNotConfigured",
                    "extendedHelp": "https://console.developers.google.com"}],
    }})


def test_disabled_api_names_the_service(api):
    hint = api._google_error_hint(_disabled_api_error())
    assert hint is not None
    assert "calendar-json.googleapis.com" in hint
    assert "is not enabled" in hint


def test_disabled_api_points_at_activation_url(api):
    hint = api._google_error_hint(_disabled_api_error())
    assert "https://console.developers.google.com" in hint


def test_insufficient_scope_says_reauthorise(api):
    hint = api._google_error_hint(_http_error(403, {"error": {
        "message": "Request had insufficient authentication scopes.",
        "errors": [{"reason": "insufficientPermissions"}],
    }}))
    assert hint is not None
    assert "--revoke" in hint


def test_expired_token(api):
    hint = api._google_error_hint(_http_error(401, {"error": {"message": "Invalid Credentials"}}))
    assert hint is not None
    assert "expired or was revoked" in hint


def test_quota_says_wait(api):
    hint = api._google_error_hint(_http_error(429, {"error": {
        "message": "Quota exceeded", "errors": [{"reason": "rateLimitExceeded"}],
    }}))
    assert hint is not None
    assert "rate-limit" in hint


def test_non_http_error_is_reraised(api):
    assert api._google_error_hint(ValueError("boom")) is None


def test_main_exits_1_without_traceback(api, monkeypatch):
    """End-to-end: the CLI prints the hint on stderr and exits 1, no traceback."""

    def _boom(_args):
        raise _disabled_api_error()

    monkeypatch.setattr(api, "gmail_search", _boom)
    monkeypatch.setattr(sys, "argv", ["google_api.py", "gmail", "search", "is:unread"])
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit) as exit_info:
        api.main()
    assert exit_info.value.code == 1
    assert "is not enabled" in err.getvalue()
    assert "Traceback" not in err.getvalue()
