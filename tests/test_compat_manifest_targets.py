"""Plugin-compat pointers must resolve to the SAME object the name moved to, never a same-named stranger.

Regression: ``hermes_cli.kanban_db.connect`` was pointed at ``hermes_cli.projects_db.connect`` (a different
database, no ``board=`` parameter) because the generator ranked candidate homes by path proximity. The
manifest codified the mistake, so the compat lint treated it as valid.

Invariant checked here: for every ``moved-lazy`` entry whose target module also exists in the manifest of
some other facade under the same name, or whose facade stem has a sibling ``<stem>_*`` module defining the
name, the facade attribute IS the sibling's object.
"""
import asyncio
import importlib
import json
import os
import pkgutil
import sqlite3
from unittest.mock import patch
from pathlib import Path

import pytest
import tools.managed_tool_gateway as managed_tool_gateway

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "compat_manifest.json"

# This file resolves every pointer on purpose; the once-per-name plugin warning is expected here.
pytestmark = [
    pytest.mark.skipif(not MANIFEST.exists(), reason="compat layer removed (scheduled revert)"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_managed_vendor_endpoints_pin_the_deployed_gateway_url():
    with patch.dict(
        os.environ,
        {"TOOL_GATEWAY_DOMAIN": "nousresearch.com", "TOOL_GATEWAY_SCHEME": "https"},
        clear=False,
    ):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        endpoints = managed_tool_gateway.managed_vendor_endpoints("vendorx")
    assert endpoints == {
        "origin": "https://tool-gateway.nousresearch.com",
        "base_url": "https://tool-gateway.nousresearch.com/api/vendorx",
        "upload_path": "/api/uploads/vendorx",
    }


def test_managed_vendor_endpoints_do_not_consult_entitlement():
    with patch.dict(os.environ, {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"}, clear=False), \
        patch.object(
            managed_tool_gateway,
            "managed_nous_tools_enabled",
            side_effect=AssertionError("entitlement must not gate address resolution"),
        ):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        endpoints = managed_tool_gateway.managed_vendor_endpoints("vendorx")
    assert endpoints is not None
    assert endpoints["base_url"] == "https://tool-gateway.nousresearch.com/api/vendorx"


def test_managed_vendor_endpoints_are_none_when_no_origin_resolves():
    with patch.dict(os.environ, {"TOOL_GATEWAY_SCHEME": "ftp"}, clear=False):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        assert managed_tool_gateway.managed_vendor_endpoints("vendorx") is None


def test_managed_vendor_endpoints_are_none_when_builder_returns_empty_origin():
    assert managed_tool_gateway.managed_vendor_endpoints(
        "vendorx", gateway_builder=lambda _vendor: ""
    ) is None


@pytest.mark.parametrize("url", [None, "", "   ", 42])
def test_managed_gateway_url_rejects_non_urls(url):
    assert managed_tool_gateway.is_managed_nous_gateway_url(url) is False


def test_managed_gateway_url_rejects_invalid_builder_url():
    assert managed_tool_gateway.is_managed_nous_gateway_url(
        "https://tool-gateway.example.com/api/vendorx",
        gateway_builder=lambda _vendor: "https://[invalid",
    ) is False


def test_managed_gateway_auth_headers_carry_the_bearer():
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        headers = managed_tool_gateway.managed_gateway_auth_headers(
            "https://tool-gateway.example.com/api/vendorx/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: "nous-token",
        )
    assert headers == {"Authorization": "Bearer nous-token"}


def test_managed_gateway_auth_headers_reflect_a_rotated_token():
    tokens = iter(["first-token", "second-token"])
    builder = lambda vendor: f"https://{vendor}-gateway.example.com"
    url = "https://tool-gateway.example.com/api/vendorx/generations"
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        first = managed_tool_gateway.managed_gateway_auth_headers(
            url, builder, lambda: next(tokens)
        )
        second = managed_tool_gateway.managed_gateway_auth_headers(
            url, builder, lambda: next(tokens)
        )
    assert first["Authorization"] == "Bearer first-token"
    assert second["Authorization"] == "Bearer second-token"


def test_managed_gateway_auth_headers_refuse_a_url_off_the_gateway_origin():
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        assert managed_tool_gateway.managed_gateway_auth_headers(
            "https://attacker.example/api/vendorx/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: "nous-token",
        ) == {}


def test_managed_gateway_auth_headers_empty_without_a_token():
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        assert managed_tool_gateway.managed_gateway_auth_headers(
            "https://tool-gateway.example.com/api/vendorx/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: None,
        ) == {}


class TestManagedMediaUploader:
    GATEWAY = "https://tool-gateway.example.com"
    BASE_URL = f"{GATEWAY}/api/vendorx"
    UPLOAD_PATH = "/api/uploads/vendorx"

    def _uploader(self, **kwargs):
        return managed_tool_gateway.build_managed_media_uploader(
            kwargs.pop("server_url", self.BASE_URL),
            kwargs.pop("upload_path", self.UPLOAD_PATH),
            gateway_builder=lambda vendor: self.GATEWAY,
            token_reader=kwargs.pop("token_reader", lambda: "nous-token"),
        )

    @staticmethod
    def _response(status_code=200, payload=None):
        class _R:
            def __init__(self):
                self.status_code = status_code

            def json(self):
                if payload is None:
                    raise ValueError("no json")
                return payload

        return _R()

    def _run(self, uploader, data=b"bytes", mime="image/png", presign=None, put=None):
        import httpx
        from tools import url_safety

        calls = {"presign": [], "put": []}
        presign = presign if presign is not None else self._response(
            200, {"uploadUrl": "https://storage.example/put?sig=abc", "token": "tok-1"}
        )
        put = put if put is not None else self._response(200)

        class _PresignClient:
            def __init__(self, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def post(self, url, headers=None, json=None):
                calls["presign"].append({"url": url, "headers": headers, "json": json})
                return presign

        class _PutClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def put(self, url, content=None, headers=None):
                calls["put"].append({"url": url, "content": content, "headers": headers})
                return put

        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True), \
            patch.object(httpx, "AsyncClient", _PresignClient), \
            patch.object(
                url_safety,
                "create_ssrf_safe_async_client",
                lambda **_kw: _PutClient(),
            ):
            calls["result"] = asyncio.run(uploader(data, mime))
        return calls

    def test_presign_declares_the_exact_type_and_length_the_put_then_sends(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        data = b"\x89PNG\r\n\x1a\n" + b"payload" * 100
        calls = self._run(uploader, data=data, mime="image/png")
        assert calls["presign"][0]["url"] == f"{self.GATEWAY}{self.UPLOAD_PATH}"
        assert calls["presign"][0]["json"] == {
            "contentType": "image/png",
            "contentLength": len(data),
        }
        assert calls["presign"][0]["headers"]["Authorization"] == "Bearer nous-token"
        assert calls["put"][0]["url"] == "https://storage.example/put?sig=abc"
        assert calls["put"][0]["content"] == data
        assert calls["put"][0]["headers"] == {"Content-Type": "image/png"}
        assert calls["result"] == "nous-upload:tok-1"

    def test_the_bytes_go_to_storage_and_never_through_the_gateway(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        calls = self._run(uploader, data=b"v" * 4096, mime="video/mp4")
        assert len(calls["presign"]) == 1 and len(calls["put"]) == 1
        assert self.GATEWAY not in calls["put"][0]["url"]
        assert calls["presign"][0]["json"]["contentType"] == "video/mp4"

    def test_no_uploader_when_the_url_is_not_a_managed_gateway(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            assert self._uploader(server_url="https://attacker.example/api/vendorx") is None

    @pytest.mark.parametrize("upload_path", [None, "", "api/uploads/vendorx", 42])
    def test_no_uploader_without_a_rooted_upload_path(self, upload_path):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            assert self._uploader(upload_path=upload_path) is None

    def test_a_missing_credential_fails_before_any_request(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True), \
            patch.object(managed_tool_gateway, "managed_gateway_auth_headers", return_value={}):
            with pytest.raises(RuntimeError, match="no Nous credential"):
                asyncio.run(uploader(b"x", "image/png"))

    def test_a_gateway_refusal_surfaces_its_own_message(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        refusal = self._response(
            413, {"error": {"message": "That file is 82MB; the limit for video is 50MB."}}
        )
        with pytest.raises(RuntimeError, match="the limit for video is 50MB"):
            self._run(uploader, presign=refusal)

    def test_an_unreadable_refusal_still_reports_the_status(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        with pytest.raises(RuntimeError, match="HTTP 502"):
            self._run(uploader, presign=self._response(502, None))

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"uploadUrl": "https://storage.example/put"},
            {"token": "tok-1"},
            {"uploadUrl": "", "token": "tok-1"},
            {"uploadUrl": "https://storage.example/put", "token": ""},
        ],
    )
    def test_a_malformed_presign_response_is_refused_rather_than_guessed(self, payload):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        with pytest.raises(RuntimeError, match="malformed"):
            self._run(uploader, presign=self._response(200, payload))

    def test_an_unreadable_success_response_is_refused_as_malformed(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        with pytest.raises(RuntimeError, match="malformed"):
            self._run(uploader, presign=self._response(200, None))

    def test_a_storage_rejection_is_not_reported_as_a_successful_upload(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        with pytest.raises(RuntimeError, match="storage refused the upload"):
            self._run(uploader, put=self._response(403))


def _entries():
    return [e for e in json.loads(MANIFEST.read_text())["entries"] if e["kind"] == "moved-lazy"]


def _sibling_modules(facade: str) -> list[str]:
    pkg, _, stem = facade.rpartition(".")
    try:
        parent = importlib.import_module(pkg) if pkg else None
    except Exception:
        return []
    paths = getattr(parent, "__path__", None) if parent else [str(ROOT)]
    if not paths:
        return []
    prefix = f"{pkg}." if pkg else ""
    return [prefix + m.name for m in pkgutil.iter_modules(paths) if m.name.startswith(stem + "_")]


def test_moved_lazy_pointers_resolve_to_the_split_off_siblings_object():
    """When a facade's own ``<stem>_*`` sibling binds the name, the facade attribute must be THAT object.

    A sibling may legitimately re-import the value from elsewhere (then the pointer target is the origin and
    the objects are identical); what must never happen is the pointer resolving to a same-named stranger.
    """
    bad = []
    for e in _entries():
        facade, name = e["facade"], e["name"]
        sibs = _sibling_modules(facade)
        if not sibs:
            continue
        try:
            got = getattr(importlib.import_module(facade), name)
        except Exception as exc:  # unresolvable pointer is its own failure
            bad.append((facade, name, f"unresolvable: {exc!r}"))
            continue
        for s in sibs:
            try:
                mod = importlib.import_module(s)
            except Exception:
                continue
            if name in vars(mod):
                sib_obj = vars(mod)[name]
                same = (sib_obj == got) if isinstance(got, (int, float, str, bytes, bool, type(None))) else (sib_obj is got)
                if not same:
                    bad.append((facade, name, e["target"], s))
    assert not bad, f"compat pointers resolve to a different object than the facade's own sibling binds: {bad}"


def test_kanban_db_connect_opens_a_kanban_board(tmp_path, monkeypatch):
    """The historical ``kanban_db.connect(board=...)`` opens a Kanban DB, not projects.db."""
    import hermes_cli.kanban_db as kanban_db
    import hermes_cli.kanban_db_connect as kanban_db_connect

    assert kanban_db.connect is kanban_db_connect.connect
    assert kanban_db.connect_closing is kanban_db_connect.connect_closing
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "board.db"
    conn = kanban_db.connect(db, board="qa")
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "tasks" in tables, tables
    assert not (tmp_path / "projects.db").exists()
    assert isinstance(sqlite3.connect(db), sqlite3.Connection)
