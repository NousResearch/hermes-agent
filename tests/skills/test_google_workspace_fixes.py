"""Regression tests for the google-workspace review fixes."""
import base64
import importlib.util
import json
import os
import stat
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "skills/productivity/google-workspace/scripts"


def _load(name, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"gw_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _b64(s):
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def test_nested_multipart_body_unpadded(monkeypatch, tmp_path):
    api = _load("google_api", monkeypatch, tmp_path)
    msg = {"payload": {"mimeType": "multipart/mixed", "parts": [
        {"mimeType": "multipart/alternative", "parts": [
            {"mimeType": "text/plain", "body": {"data": _b64("Bonjour é")}},
            {"mimeType": "text/html", "body": {"data": _b64("<p>x</p>")}},
        ]},
        {"mimeType": "text/plain", "filename": "a.txt", "body": {"attachmentId": "z"}},
    ]}}
    assert api._extract_message_body(msg) == "Bonjour é"
    html_only = {"payload": {"parts": [{"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/html", "body": {"data": _b64("<b>h</b>")}}]}]}}
    assert api._extract_message_body(html_only) == "<b>h</b>"
    simple = {"payload": {"body": {"data": _b64("plain")}}}
    assert api._extract_message_body(simple) == "plain"


def _reply_args(**kw):
    return types.SimpleNamespace(body="ok", from_header=None, **kw)


def test_reply_honours_reply_to_and_references(monkeypatch, tmp_path):
    api = _load("google_api", monkeypatch, tmp_path)
    original = {"labelIds": ["INBOX"], "payload": {"headers": [
        {"name": "From", "value": "a@x.com"},
        {"name": "Reply-To", "value": "list@x.com"},
        {"name": "Subject", "value": "RE: plan"},
        {"name": "Message-ID", "value": "<m2@x>"},
        {"name": "References", "value": "<m1@x>"},
    ]}}
    m = api._build_reply_message(original, _reply_args())
    assert m["To"] == "list@x.com"
    assert m["Subject"] == "RE: plan"
    assert m["References"] == "<m1@x> <m2@x>"
    assert m["In-Reply-To"] == "<m2@x>"


def test_reply_to_own_sent_message_targets_recipients(monkeypatch, tmp_path):
    api = _load("google_api", monkeypatch, tmp_path)
    original = {"labelIds": ["SENT"], "payload": {"headers": [
        {"name": "From", "value": "me@x.com"},
        {"name": "To", "value": "bob@y.com"},
        {"name": "Subject", "value": "plan"},
    ]}}
    m = api._build_reply_message(original, _reply_args())
    assert m["To"] == "bob@y.com" and m["Subject"] == "Re: plan"


class _FakeDrive:
    def __init__(self, name, mime="application/pdf"):
        self.meta = {"name": name, "mimeType": mime}
        self.captured = {}

    def files(self):
        return self

    def get(self, **kw):
        return types.SimpleNamespace(execute=lambda: self.meta)

    def list(self, **kw):
        self.captured.update(kw)
        return types.SimpleNamespace(execute=lambda: {"files": []})


def test_drive_search_escapes_query(monkeypatch, tmp_path, capsys):
    api = _load("google_api", monkeypatch, tmp_path)
    fake = _FakeDrive("x")
    monkeypatch.setattr(api, "_gws_binary", lambda: None)
    monkeypatch.setattr(api, "build_service", lambda *a, **k: fake)
    api.drive_search(types.SimpleNamespace(query="O'Brien\\", raw_query=False, max=5))
    assert fake.captured["q"] == "fullText contains 'O\\'Brien\\\\' and trashed = false"


def test_drive_download_name_cannot_escape_cwd(monkeypatch, tmp_path):
    api = _load("google_api", monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    fake = _FakeDrive("../../evil.sh")
    written = {}

    class _Req:
        pass

    fake.get_media = lambda **kw: _Req()

    class _Dl:
        def __init__(self, fh, req, **kw):
            written["path"] = Path(fh.name).resolve()
            fh.write(b"x")

        def next_chunk(self):
            return None, True

    monkeypatch.setattr(api, "_gws_binary", lambda: None)
    monkeypatch.setattr(api, "build_service", lambda *a, **k: fake)
    sys.modules.setdefault("googleapiclient", types.ModuleType("googleapiclient"))
    http = types.ModuleType("googleapiclient.http")
    http.MediaIoBaseDownload = _Dl
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)
    api.drive_download(types.SimpleNamespace(file_id="F1", output=None, export_mime=None))
    assert written["path"] == (work / "evil.sh").resolve()
    assert not (tmp_path / "evil.sh").exists()


@pytest.mark.parametrize("content,expected", [
    ([{"endIndex": 1}, {"endIndex": 5}], "\nnew"),   # "old\n"
    ([{"endIndex": 1}, {"endIndex": 2}], "new"),      # empty doc
])
def test_docs_append_starts_new_paragraph(monkeypatch, tmp_path, capsys, content, expected):
    api = _load("google_api", monkeypatch, tmp_path)
    sent = {}
    monkeypatch.setattr(api, "_gws_binary", lambda: None)
    monkeypatch.setattr(api, "_resolve_write_tab", lambda doc, tab: (None, {"content": content}))
    monkeypatch.setattr(api, "_docs_insert_text", lambda doc_id, text, index, tab_id=None: sent.update(text=text, index=index))

    class _Docs:
        def documents(self):
            return self

        def get(self, **kw):
            return types.SimpleNamespace(execute=dict)

    monkeypatch.setattr(api, "build_service", lambda *a, **k: _Docs())
    api.docs_append(types.SimpleNamespace(doc_id="D", text="new\n", tab=None))
    assert sent["text"] == expected


def test_bridge_refreshes_naive_and_near_expiry(monkeypatch, tmp_path):
    br = _load("gws_bridge", monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(br, "refresh_token", lambda d: calls.append(1) or {**d, "token": "new"})
    tok = tmp_path / "google_token.json"
    naive_past = (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None).isoformat()
    tok.write_text(json.dumps({"token": "old", "expiry": naive_past}))
    assert br.get_valid_token() == "new"
    soon = (datetime.now(UTC) + timedelta(seconds=20)).isoformat()
    tok.write_text(json.dumps({"token": "old", "expiry": soon}))
    assert br.get_valid_token() == "new"
    later = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    tok.write_text(json.dumps({"token": "old", "expiry": later}))
    assert br.get_valid_token() == "old"
    assert len(calls) == 2


def test_revoke_keeps_token_on_network_failure(monkeypatch, tmp_path):
    setup = _load("setup", monkeypatch, tmp_path)
    tok = tmp_path / "google_token.json"
    tok.write_text(json.dumps({
        "type": "authorized_user", "token": "AT", "refresh_token": "RT",
        "client_id": "c", "client_secret": "s", "token_uri": "https://oauth2.googleapis.com/token",
        "expiry": (datetime.now(UTC) + timedelta(hours=1)).replace(tzinfo=None).isoformat(),
    }))
    monkeypatch.setattr(setup, "_ensure_deps", lambda: None)
    import urllib.request
    seen = {}

    def boom(req, timeout=None):
        seen["url"], seen["data"] = req.full_url, req.data
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert setup.revoke() is False
    assert tok.exists()
    assert seen["url"] == "https://oauth2.googleapis.com/revoke"
    assert seen["data"] == b"token=RT"


def test_revoke_cleans_up_expired_revoked_token(monkeypatch, tmp_path):
    """An expired token whose refresh token is dead must still be removable."""
    setup = _load("setup", monkeypatch, tmp_path)
    tok = tmp_path / "google_token.json"
    tok.write_text(json.dumps({
        "type": "authorized_user", "token": "AT", "refresh_token": "RT",
        "client_id": "c", "client_secret": "s", "token_uri": "https://oauth2.googleapis.com/token",
        "expiry": (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None).isoformat(),
    }))
    monkeypatch.setattr(setup, "_ensure_deps", lambda: None)
    import io
    import urllib.error
    import urllib.request

    from google.oauth2.credentials import Credentials

    def no_refresh(self, request):
        raise AssertionError("revoke must not refresh first")

    monkeypatch.setattr(Credentials, "refresh", no_refresh)

    def already_invalid(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "invalid_token", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(urllib.request, "urlopen", already_invalid)
    assert setup.revoke() is not False
    assert not tok.exists()


@pytest.mark.parametrize("subject,expected", [
    ("Re : plan", "Re : plan"),
    ("re:plan", "re:plan"),
    ("Repas", "Re: Repas"),
])
def test_reply_subject_prefix_detection(monkeypatch, tmp_path, subject, expected):
    api = _load("google_api", monkeypatch, tmp_path)
    original = {"labelIds": ["INBOX"], "payload": {"headers": [
        {"name": "From", "value": "a@x.com"}, {"name": "Subject", "value": subject}]}}
    assert api._build_reply_message(original, _reply_args())["Subject"] == expected


def test_docs_append_rejects_empty_text(monkeypatch, tmp_path):
    api = _load("google_api", monkeypatch, tmp_path)
    calls = []

    class _Docs:
        def documents(self):
            return self

        def get(self, **kw):
            return types.SimpleNamespace(execute=lambda: {"body": {"content": [{"endIndex": 1}]}})

    monkeypatch.setattr(api, "build_service", lambda *a, **k: _Docs())
    monkeypatch.setattr(api, "_docs_insert_text", lambda *a, **k: calls.append(a))
    with pytest.raises(SystemExit) as e:
        api.docs_append(types.SimpleNamespace(doc_id="D", text="\n\n", tab=None))
    assert e.value.code == 1 and not calls


def test_drive_download_does_not_follow_default_name_symlink(monkeypatch, tmp_path):
    """An untrusted Drive name must not overwrite a file through a cwd symlink."""
    api = _load("google_api", monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    try:
        (work / "report.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")

    fake = _FakeDrive("report.txt")
    fake.get_media = lambda **_kw: object()
    monkeypatch.chdir(work)
    monkeypatch.setattr(api, "build_service", lambda *_a, **_kw: fake)
    http = types.ModuleType("googleapiclient.http")
    http.MediaIoBaseDownload = lambda *_a, **_kw: pytest.fail("must refuse before download")
    monkeypatch.setitem(sys.modules, "googleapiclient", types.ModuleType("googleapiclient"))
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)

    with pytest.raises(SystemExit) as exc:
        api.drive_download(types.SimpleNamespace(file_id="F1", output=None, export_mime=None))
    assert exc.value.code == 1
    assert outside.read_text(encoding="utf-8") == "keep"
