"""Offline tests for optional-skills/email/cooper-email/scripts/cooper_mail.py."""
import importlib.util
import io
import json
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "optional-skills" / "email" / "cooper-email" / "scripts" / "cooper_mail.py"


def _load():
    spec = importlib.util.spec_from_file_location("cooper_mail", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture(mod, argv, env, payload=b'{"object":"x"}'):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["req"] = req
        return _Resp(payload)

    with mock.patch.dict("os.environ", env, clear=True), \
         mock.patch.object(mod.urllib.request, "urlopen", fake_urlopen), \
         mock.patch("builtins.print") as out:
        code = mod.main(argv)
    return code, seen.get("req"), json.loads(out.call_args[0][0])


def test_onboard_sends_no_auth_header():
    mod = _load()
    code, req, out = _capture(mod, ["onboard", "bot-7f3a", "--owner-email", "h@example.com"], {})
    assert code == 0 and out["ok"] is True
    assert req.get_method() == "POST" and req.full_url.endswith("/api/v1/onboard")
    assert req.get_header("Authorization") is None
    assert json.loads(req.data) == {"username": "bot-7f3a", "owner_email": "h@example.com"}


def test_send_uses_bearer_and_idempotency_key():
    mod = _load()
    env = {"COOPER_API_KEY": "coop_live_test", "COOPER_INBOX_ID": "inb_1"}
    code, req, out = _capture(
        mod, ["send", "--to", "a@example.com", "--subject", "Hi", "--text", "x", "--idempotency-key", "k1"], env)
    assert code == 0
    assert req.full_url.endswith("/api/v1/inboxes/inb_1/messages")
    assert req.get_header("Authorization") == "Bearer coop_live_test"
    assert req.get_header("Idempotency-key") == "k1"
    assert "coop_live_test" not in json.dumps(out)


def test_wait_is_clamped_to_55_seconds():
    mod = _load()
    env = {"COOPER_API_KEY": "coop_live_test", "COOPER_INBOX_ID": "inb_1"}
    _, req, _ = _capture(mod, ["wait-code", "--wait", "300", "--from", "example.com"], env)
    assert "/inboxes/inb_1/wait-for-code?" in req.full_url
    assert "wait=55" in req.full_url and "from=example.com" in req.full_url


def test_missing_key_fails_without_network():
    mod = _load()
    with mock.patch.dict("os.environ", {}, clear=True), \
         mock.patch.object(mod.urllib.request, "urlopen") as net:
        with pytest.raises(SystemExit):
            mod.main(["me"])
        net.assert_not_called()
