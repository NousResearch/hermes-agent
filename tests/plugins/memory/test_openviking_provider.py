import json
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.openviking import OpenVikingMemoryProvider, _VikingClient


def test_tool_search_sorts_by_raw_score_across_buckets():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/1", "score": 0.9003, "abstract": "memory result"},
            ],
            "resources": [
                {"uri": "viking://resources/1", "score": 0.9004, "abstract": "resource result"},
            ],
            "skills": [
                {"uri": "viking://skills/1", "score": 0.8999, "abstract": "skill result"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://resources/1",
        "viking://memories/1",
        "viking://skills/1",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.9, 0.9, 0.9]
    assert result["total"] == 3


def test_tool_search_sorts_missing_raw_score_after_negative_scores():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "result": {
            "memories": [
                {"uri": "viking://memories/missing", "abstract": "missing score"},
            ],
            "resources": [
                {"uri": "viking://resources/negative", "score": -0.25, "abstract": "negative score"},
            ],
            "skills": [
                {"uri": "viking://skills/positive", "score": 0.1, "abstract": "positive score"},
            ],
            "total": 3,
        }
    }

    result = json.loads(provider._tool_search({"query": "ranking"}))

    assert [entry["uri"] for entry in result["results"]] == [
        "viking://skills/positive",
        "viking://memories/missing",
        "viking://resources/negative",
    ]
    assert [entry["score"] for entry in result["results"]] == [0.1, 0.0, -0.25]
    assert result["total"] == 3


def test_tool_add_resource_uploads_existing_local_file(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.upload_temp_file.return_value = "upload_sample.md"
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/sample"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": str(sample),
        "reason": "local test",
        "wait": True,
    }))

    provider._client.upload_temp_file.assert_called_once_with(sample)
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "local test",
        "wait": True,
        "source_name": "sample.md",
        "temp_file_id": "upload_sample.md",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/sample"


def test_tool_add_resource_uploads_file_uri(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.upload_temp_file.return_value = "upload_sample.md"
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/sample"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": sample.as_uri(),
        "reason": "file uri test",
    }))

    provider._client.upload_temp_file.assert_called_once_with(sample)
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "file uri test",
        "source_name": "sample.md",
        "temp_file_id": "upload_sample.md",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/sample"


def test_tool_add_resource_uploads_existing_local_directory_and_cleans_zip(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    nested = docs / "nested"
    nested.mkdir()
    (nested / "api.md").write_text("# API\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []
    provider._client.upload_temp_file.side_effect = (
        lambda path: uploaded_paths.append(path) or "upload_docs.zip"
    )
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/docs"},
    }

    result = json.loads(provider._tool_add_resource({
        "url": str(docs),
        "reason": "directory test",
        "wait": True,
    }))

    assert uploaded_paths
    assert uploaded_paths[0].suffix == ".zip"
    assert not uploaded_paths[0].exists()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "reason": "directory test",
        "wait": True,
        "source_name": "docs",
        "temp_file_id": "upload_docs.zip",
    })
    assert result["status"] == "added"
    assert result["root_uri"] == "viking://resources/docs"


def test_tool_add_resource_directory_zip_skips_symlink_escape(tmp_path):
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("do not upload\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    link = docs / "leak.txt"
    try:
        link.symlink_to(secret)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    archive_entries = {}

    def inspect_upload(path):
        with zipfile.ZipFile(path) as archive:
            archive_entries["names"] = archive.namelist()
            archive_entries["payloads"] = {
                name: archive.read(name)
                for name in archive.namelist()
            }
        return "upload_docs.zip"

    provider._client.upload_temp_file.side_effect = inspect_upload
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/docs"},
    }

    json.loads(provider._tool_add_resource({"url": str(docs)}))

    assert archive_entries["names"] == ["guide.md"]
    assert b"do not upload" not in b"".join(archive_entries["payloads"].values())


def test_tool_add_resource_cleans_local_directory_zip_when_add_fails(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []
    provider._client.upload_temp_file.side_effect = (
        lambda path: uploaded_paths.append(path) or "upload_docs.zip"
    )
    provider._client.post.side_effect = RuntimeError("add failed")

    with pytest.raises(RuntimeError, match="add failed"):
        provider._tool_add_resource({"url": str(docs)})

    assert uploaded_paths
    assert not uploaded_paths[0].exists()


def test_tool_add_resource_cleans_local_directory_zip_when_upload_fails(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    uploaded_paths = []

    def fail_upload(path):
        uploaded_paths.append(path)
        raise RuntimeError("upload failed")

    provider._client.upload_temp_file.side_effect = fail_upload

    with pytest.raises(RuntimeError, match="upload failed"):
        provider._tool_add_resource({"url": str(docs)})

    assert uploaded_paths
    assert not uploaded_paths[0].exists()
    provider._client.post.assert_not_called()


def test_tool_add_resource_rejects_missing_local_path(tmp_path):
    missing = tmp_path / "missing.md"
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()

    result = json.loads(provider._tool_add_resource({"url": str(missing)}))

    assert result["error"] == f"Local resource path does not exist: {missing}"
    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_not_called()


def test_tool_add_resource_sends_remote_url_as_path():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/remote"},
    }

    provider._tool_add_resource({"url": "https://example.com/doc.md"})

    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "path": "https://example.com/doc.md",
    })


@pytest.mark.parametrize("url", [
    "git@github.com:org/repo.git",
    "git@ssh.dev.azure.com:v3/org/project/repo",
    "ssh://git@github.com/org/repo.git",
    "git://github.com/org/repo.git",
])
def test_tool_add_resource_sends_git_remote_sources_as_path(url):
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._client.post.return_value = {
        "status": "ok",
        "result": {"root_uri": "viking://resources/repo"},
    }

    provider._tool_add_resource({"url": url})

    provider._client.upload_temp_file.assert_not_called()
    provider._client.post.assert_called_once_with("/api/v1/resources", {
        "path": url,
    })


def test_viking_client_upload_temp_file_uses_multipart_identity_headers(tmp_path, monkeypatch):
    sample = tmp_path / "sample.md"
    sample.write_text("# Local resource\n", encoding="utf-8")
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="test-account",
        user="test-user",
        agent="test-agent",
    )
    captured_kwargs = {}

    def capture_httpx_post(url, **kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"status": "ok", "result": {"temp_file_id": "upload_sample.md"}},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(client._httpx, "post", capture_httpx_post)

    assert client.upload_temp_file(sample) == "upload_sample.md"

    assert "files" in captured_kwargs
    assert "json" not in captured_kwargs
    headers = captured_kwargs["headers"]
    assert headers["X-OpenViking-Account"] == "test-account"
    assert headers["X-OpenViking-User"] == "test-user"
    assert headers["X-OpenViking-Agent"] == "test-agent"
    assert headers["X-API-Key"] == "test-key"
    assert "Content-Type" not in headers


def test_viking_client_raises_structured_server_error():
    client = _VikingClient.__new__(_VikingClient)
    response = SimpleNamespace(
        status_code=403,
        text='{"status":"error"}',
        json=lambda: {
            "status": "error",
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "direct host filesystem paths are not allowed",
            },
        },
        raise_for_status=lambda: None,
    )

    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        client._parse_response(response)


def test_viking_client_headers_include_bearer_when_api_key_set():
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="acct",
        user="usr",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-API-Key"] == "test-key"
    assert headers["Authorization"] == "Bearer test-key"


def test_viking_client_headers_send_tenant_when_default():
    # account/user set to the literal string "default". OpenViking 0.3.x
    # requires X-OpenViking-Account and X-OpenViking-User for ROOT API key
    # requests to tenant-scoped APIs — omitting them causes
    # INVALID_ARGUMENT errors even when account="default".
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="default",
        user="default",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-OpenViking-Account"] == "default"
    assert headers["X-OpenViking-User"] == "default"
    assert headers["X-OpenViking-Agent"] == "hermes"
    assert headers["Authorization"] == "Bearer test-key"


def test_viking_client_headers_send_tenant_when_empty_falls_back_to_default():
    # Empty account/user strings fall back to "default" via the constructor.
    # Headers are sent even for the default value — ROOT API keys need them.
    client = _VikingClient(
        "https://example.com",
        api_key="",
        account="",
        user="",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-OpenViking-Account"] == "default"
    assert headers["X-OpenViking-User"] == "default"
    assert "Authorization" not in headers
    assert "X-API-Key" not in headers


def test_viking_client_headers_sent_with_real_tenant_values():
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="real-account",
        user="real-user",
        agent="hermes",
    )
    headers = client._headers()
    assert headers["X-OpenViking-Account"] == "real-account"
    assert headers["X-OpenViking-User"] == "real-user"


def test_viking_client_health_sends_auth_headers(monkeypatch):
    client = _VikingClient(
        "https://example.com",
        api_key="test-key",
        account="",
        user="",
        agent="hermes",
    )
    captured = {}

    def capture_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(client._httpx, "get", capture_get)
    assert client.health() is True
    assert captured["url"] == "https://example.com/health"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


# ---------------------------------------------------------------------------
# on_session_switch — flush + commit + rotate behavior (hermes-agent#28296)
# ---------------------------------------------------------------------------

def _make_provider_with_session(session_id: str, turn_count: int):
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._session_id = session_id
    provider._turn_count = turn_count
    return provider


def test_on_session_switch_commits_old_session_and_rotates_id():
    provider = _make_provider_with_session("old-sid", turn_count=3)

    provider.on_session_switch("new-sid", parent_session_id="old-sid")

    provider._client.post.assert_called_once_with("/api/v1/sessions/old-sid/commit")
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_on_session_switch_skips_commit_for_empty_old_session():
    """No turns accumulated → nothing to extract → no commit call."""
    provider = _make_provider_with_session("old-sid", turn_count=0)

    provider.on_session_switch("new-sid")

    provider._client.post.assert_not_called()
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_on_session_switch_clears_stale_prefetch_result():
    provider = _make_provider_with_session("old-sid", turn_count=1)
    provider._prefetch_result = "stale recall from old session"

    provider.on_session_switch("new-sid")

    assert provider._prefetch_result == ""


def test_on_session_switch_waits_for_inflight_sync_thread():
    """In-flight sync_turn write must drain before the commit fires —
    otherwise the commit can race the last message write."""
    provider = _make_provider_with_session("old-sid", turn_count=2)

    join_calls = []

    class FakeThread:
        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            join_calls.append(timeout)
            # Simulate a worker that finishes within the join window.
            self._alive = False

    provider._inflight_writers["old-sid"] = {FakeThread()}

    provider.on_session_switch("new-sid")

    assert join_calls, "expected on_session_switch to join the in-flight sync thread"
    provider._client.post.assert_called_once_with("/api/v1/sessions/old-sid/commit")


def test_on_session_switch_noop_on_empty_new_id():
    provider = _make_provider_with_session("old-sid", turn_count=5)

    provider.on_session_switch("")
    provider.on_session_switch("   ")

    provider._client.post.assert_not_called()
    assert provider._session_id == "old-sid"
    assert provider._turn_count == 5


def test_on_session_switch_noop_when_client_missing():
    provider = OpenVikingMemoryProvider()
    provider._client = None
    provider._session_id = "old-sid"
    provider._turn_count = 4

    # Must not raise even though no client is configured.
    provider.on_session_switch("new-sid")

    # State stays untouched — provider is effectively disabled.
    assert provider._session_id == "old-sid"
    assert provider._turn_count == 4


def test_sync_turn_captures_session_id_before_worker_runs():
    """Worker must use the session id snapshotted at sync_turn() call time, not
    re-read self._session_id later — otherwise a delayed worker can write the
    previous turn's messages into the rotated-in NEW session."""
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "old-sid"

    started = threading.Event()
    release = threading.Event()
    captured_paths = []

    def fake_post(path, payload=None, **kwargs):
        started.set()
        release.wait(timeout=2.0)
        captured_paths.append(path)
        return {}

    # Patch _VikingClient inside the worker by stubbing post on a client
    # the constructor will produce. Easiest path: monkeypatch the class.
    real_client_cls = _VikingClient

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            return fake_post(path, payload, **kwargs)

    import plugins.memory.openviking as _mod
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u", "a")
        # Wait until the worker is parked inside the first post call.
        assert started.wait(timeout=2.0), "worker never entered post()"
        # Rotate the provider's session id while the worker is mid-flight.
        provider._session_id = "new-sid"
        release.set()
        for t in list(provider._inflight_writers.get("old-sid", set())):
            t.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    # Both writes must target the OLD session id captured at call time.
    assert captured_paths == [
        "/api/v1/sessions/old-sid/messages",
        "/api/v1/sessions/old-sid/messages",
    ]


def test_sync_turn_noop_when_session_id_blank():
    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._session_id = ""

    provider.sync_turn("u", "a")

    # No turn counted, no worker spawned.
    assert provider._turn_count == 0
    assert provider._inflight_writers == {}


def test_on_session_end_marks_session_clean_after_successful_commit():
    """After a successful commit on_session_end must reset _turn_count so a
    subsequent on_session_switch (fired by /new and compression right after
    commit_memory_session) skips its commit instead of double-committing."""
    provider = _make_provider_with_session("old-sid", turn_count=3)

    provider.on_session_end([])

    provider._client.post.assert_called_once_with("/api/v1/sessions/old-sid/commit")
    assert provider._turn_count == 0


def test_on_session_end_keeps_dirty_when_commit_fails():
    """If the commit fails, leave _turn_count > 0 so on_session_switch retries
    rather than silently dropping extraction for the old session."""
    provider = _make_provider_with_session("old-sid", turn_count=3)
    provider._client.post.side_effect = RuntimeError("commit boom")

    provider.on_session_end([])

    assert provider._turn_count == 3


def test_end_then_switch_does_not_double_commit():
    """Mirrors the /new and compression call order: commit_memory_session
    (→ on_session_end) immediately followed by on_session_switch. The switch
    must NOT issue a second commit on the same session id."""
    provider = _make_provider_with_session("old-sid", turn_count=2)

    provider.on_session_end([])
    provider.on_session_switch("new-sid", parent_session_id="old-sid")

    # Exactly one commit call, on the OLD session, fired by on_session_end.
    provider._client.post.assert_called_once_with("/api/v1/sessions/old-sid/commit")
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_on_session_switch_swallows_commit_failure():
    """Commit-on-switch must not propagate exceptions: a failing commit on the
    old session must still allow the rotate to the new session to complete,
    otherwise subsequent sync_turn writes would land in the wrong session."""
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._client.post.side_effect = RuntimeError("commit boom")

    provider.on_session_switch("new-sid")

    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


# ---------------------------------------------------------------------------
# Hung-writer protection: the sync worker can outlive the bounded join
# because each OpenViking POST has _TIMEOUT=30s and there are two per turn.
# Committing while late writes are still in flight would orphan them past
# the commit boundary — they would never be extracted.
# ---------------------------------------------------------------------------

class _HungThread:
    """Thread stand-in that stays alive across joins."""

    def is_alive(self):
        return True

    def join(self, timeout=None):
        # Pretend the join timed out — worker still running.
        return None


def test_on_session_end_skips_commit_when_sync_worker_outlives_join():
    """If the sync worker is still alive after the 10s join, the commit must
    be skipped — late writes from the worker would otherwise land in an
    already-committed session and never be extracted. Leave _turn_count
    intact so the session stays marked dirty."""
    provider = _make_provider_with_session("old-sid", turn_count=3)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_end([])

    provider._client.post.assert_not_called()
    assert provider._turn_count == 3


def test_on_session_switch_skips_commit_when_sync_worker_outlives_join():
    """Same hazard on the switch path. Rotation must still proceed (the new
    session needs to start) but the old-session commit is skipped to avoid
    orphaning the worker's late writes past commit."""
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_switch("new-sid")

    provider._client.post.assert_not_called()
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


# ---------------------------------------------------------------------------
# Orphaned-writer hazard: commit must wait for ALL writers for the session,
# not just the latest tracked one. sync_turn's bounded rate-limit can drop a
# still-alive previous worker — that dropped writer keeps POSTing under the
# old sid and would otherwise land its writes past the commit boundary.
# ---------------------------------------------------------------------------

def test_on_session_end_waits_for_all_writers_not_just_latest():
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_end([])

    provider._client.post.assert_not_called()
    assert provider._turn_count == 2


def test_on_session_switch_waits_for_all_writers_not_just_latest():
    provider = _make_provider_with_session("old-sid", turn_count=2)
    provider._inflight_writers["old-sid"] = {_HungThread()}

    provider.on_session_switch("new-sid")

    provider._client.post.assert_not_called()
    assert provider._session_id == "new-sid"
    assert provider._turn_count == 0


def test_sync_turn_tracks_writer_under_session_id():
    """Every sync_turn writer must register under its captured sid so the
    drain at end/switch sees it even if a later sync_turn replaces the
    latest-tracked reference."""
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "sid-1"

    release = threading.Event()
    started = threading.Event()

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            started.set()
            release.wait(timeout=2.0)
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.sync_turn("u", "a")
        assert started.wait(timeout=2.0), "worker never entered post()"
        assert len(provider._inflight_writers.get("sid-1", set())) == 1
        release.set()
        for t in list(provider._inflight_writers.get("sid-1", set())):
            t.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    # Worker should have removed itself from the inflight set on exit.
    assert provider._inflight_writers.get("sid-1", set()) == set()


# ---------------------------------------------------------------------------
# on_memory_write: explicit memory writes use content/write and stay outside
# the session transcript/commit boundary.
# ---------------------------------------------------------------------------

def test_on_memory_write_uses_content_write_independent_of_session_rotation():
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "old-sid"

    in_ctor = threading.Event()
    release = threading.Event()
    done = threading.Event()
    captured_paths = []
    captured_payloads = []

    class StubClient:
        def __init__(self, *a, **kw):
            in_ctor.set()
            release.wait(timeout=2.0)

        def post(self, path, payload=None, **kwargs):
            captured_paths.append(path)
            captured_payloads.append(payload)
            done.set()
            return {}

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.on_memory_write("add", "user", "remember this")
        assert in_ctor.wait(timeout=2.0), "worker never entered ctor"
        # Rotate provider's session id while the worker is parked. Memory writes
        # must not become session messages in either the old or new session.
        provider._session_id = "new-sid"
        release.set()
        assert done.wait(timeout=2.0), "worker never reached post()"
    finally:
        _mod._VikingClient = real_client_cls

    assert captured_paths == ["/api/v1/content/write"]
    assert captured_payloads[0]["content"] == "remember this"
    assert captured_payloads[0]["mode"] == "create"
    assert captured_payloads[0]["uri"].startswith(
        "viking://user/usr/agent/hermes/memories/preferences/mem_"
    )


# ---------------------------------------------------------------------------
# Prefetch staleness: a prefetch worker that finishes AFTER a session switch
# must drop its result instead of repopulating the new session with stale
# recall from the old generation. Bump the generation directly (rather than
# calling on_session_switch, whose own join blocks on the test worker) so
# the test isolates the generation-gating behavior.
# ---------------------------------------------------------------------------

def test_queue_prefetch_drops_result_when_generation_changed_mid_flight():
    import threading

    provider = OpenVikingMemoryProvider()
    provider._client = MagicMock()
    provider._endpoint = "http://test"
    provider._api_key = ""
    provider._account = "acct"
    provider._user = "usr"
    provider._agent = "hermes"
    provider._session_id = "old-sid"

    started = threading.Event()
    release = threading.Event()

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def post(self, path, payload=None, **kwargs):
            started.set()
            release.wait(timeout=2.0)
            return {
                "result": {
                    "memories": [
                        {"uri": "viking://memories/old", "score": 0.9,
                         "abstract": "stale from old session"},
                    ],
                    "resources": [],
                }
            }

    import plugins.memory.openviking as _mod
    real_client_cls = _mod._VikingClient
    _mod._VikingClient = StubClient
    try:
        provider.queue_prefetch("anything")
        assert started.wait(timeout=2.0), "prefetch worker never entered post()"
        # Simulate a session switch by bumping the generation directly.
        # The worker captured the pre-bump generation when it was spawned.
        provider._prefetch_generation += 1
        release.set()
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)
    finally:
        _mod._VikingClient = real_client_cls

    # The stale result from the pre-bump generation must NOT have been written
    # into the new generation's prefetch slot.
    assert provider._prefetch_result == ""
