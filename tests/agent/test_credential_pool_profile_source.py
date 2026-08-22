"""Profile fallback pools persist mutations to the store that owns them."""

from __future__ import annotations

import json
import multiprocessing
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED, load_pool
from hermes_cli import auth as auth_mod


PROVIDER = "openai-codex"


class _TokenHandler(BaseHTTPRequestHandler):
    calls = 0
    calls_lock = threading.Lock()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        refresh_token = parse_qs(body).get("refresh_token", [""])[0]
        with type(self).calls_lock:
            type(self).calls += 1
        if refresh_token == "refresh-old":
            access_token = "access-process-rotated"
            rotated_refresh_token = "refresh-process-rotated"
        else:
            access_token = f"access-{refresh_token}-rotated"
            rotated_refresh_token = f"{refresh_token}-rotated"
        payload = json.dumps({
            "access_token": access_token,
            "refresh_token": rotated_refresh_token,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return


def _refresh_in_process(
    profile_auth: str,
    root_auth: str,
    token_url: str,
    credential_id: str,
    ready_queue,
    start_event,
    result_queue,
) -> None:
    from pathlib import Path

    from agent import credential_pool as pool_mod
    from hermes_cli import auth as process_auth

    profile_path = Path(profile_auth)
    root_path = Path(root_auth)
    process_auth._auth_file_path = lambda: profile_path
    process_auth._global_auth_file_path = lambda: root_path
    pool_mod._global_auth_file_path = lambda: root_path
    process_auth.CODEX_OAUTH_TOKEN_URL = token_url
    pool = pool_mod.load_pool(PROVIDER)
    entry = next(item for item in pool.entries() if item.id == credential_id)
    ready_queue.put("ready")
    if not start_event.wait(timeout=10):
        result_queue.put(("error", "start timeout"))
        return
    try:
        refreshed = pool._refresh_entry(entry, force=True)
        result_queue.put(("ok", refreshed.refresh_token if refreshed else None))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _entry(
    credential_id: str = "global-codex",
    *,
    access_token: str = "access-old",
    refresh_token: str = "refresh-old",
) -> dict:
    return {
        "id": credential_id,
        "label": credential_id,
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


def _write_store(
    path: Path,
    entries: list[dict],
    *,
    providers: dict | None = None,
    suppressed_sources: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "version": 1,
            "providers": providers or {},
            "credential_pool": {PROVIDER: entries},
            "suppressed_sources": suppressed_sources or {},
        }),
        encoding="utf-8",
    )


def _read_store(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def profile_pool_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    root_auth = root / "auth.json"
    profile_auth = profile / "auth.json"
    _write_store(root_auth, [_entry()])
    return root_auth, profile_auth


def test_fallback_pool_status_persists_to_global_owner_without_profile_shadow(
    profile_pool_env,
):
    root_auth, profile_auth = profile_pool_env
    pool = load_pool(PROVIDER)
    entry = pool.entries()[0]

    pool._mark_exhausted(entry, 429)

    root_entry = _read_store(root_auth)["credential_pool"][PROVIDER][0]
    assert root_entry["last_status"] == STATUS_EXHAUSTED
    assert not profile_auth.exists()


def test_fallback_pool_remove_targets_global_owner(profile_pool_env):
    root_auth, profile_auth = profile_pool_env
    pool = load_pool(PROVIDER)

    removed = pool.remove_index(1)

    assert removed is not None
    assert _read_store(root_auth)["credential_pool"][PROVIDER] == []
    assert not profile_auth.exists()


def test_profile_owned_pool_still_shadows_and_writes_only_profile(
    profile_pool_env,
):
    root_auth, profile_auth = profile_pool_env
    _write_store(profile_auth, [_entry("profile-codex")])
    pool = load_pool(PROVIDER)

    pool._mark_exhausted(pool.entries()[0], 429)

    root_entry = _read_store(root_auth)["credential_pool"][PROVIDER][0]
    profile_entry = _read_store(profile_auth)["credential_pool"][PROVIDER][0]
    assert root_entry.get("last_status") is None
    assert profile_entry["id"] == "profile-codex"
    assert profile_entry["last_status"] == STATUS_EXHAUSTED


def test_explicit_profile_add_still_creates_local_shadow(profile_pool_env):
    root_auth, profile_auth = profile_pool_env
    pool = load_pool(PROVIDER)

    pool.add_entry(
        PooledCredential.from_dict(
            PROVIDER,
            _entry(
                "profile-personal",
                access_token="profile-access",
                refresh_token="profile-refresh",
            ),
        )
    )

    assert [
        entry["id"] for entry in _read_store(root_auth)["credential_pool"][PROVIDER]
    ] == ["global-codex"]
    assert [
        entry["id"] for entry in _read_store(profile_auth)["credential_pool"][PROVIDER]
    ] == ["global-codex", "profile-personal"]


def test_stale_fallback_pool_instances_refresh_single_use_token_once(
    profile_pool_env,
    monkeypatch,
):
    root_auth, profile_auth = profile_pool_env
    first_pool = load_pool(PROVIDER)
    second_pool = load_pool(PROVIDER)
    refresh_calls: list[str] = []

    def fake_refresh(access_token, refresh_token, **_kwargs):
        refresh_calls.append(refresh_token)
        return {
            "access_token": "access-rotated",
            "refresh_token": "refresh-rotated",
            "last_refresh": "2026-08-20T17:00:00Z",
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake_refresh)

    first = first_pool._refresh_entry(first_pool.entries()[0], force=True)
    second = second_pool._refresh_entry(second_pool.entries()[0], force=True)

    assert first is not None
    assert second is not None
    assert refresh_calls == ["refresh-old"]
    assert second.refresh_token == "refresh-rotated"
    root_entry = _read_store(root_auth)["credential_pool"][PROVIDER][0]
    assert root_entry["access_token"] == "access-rotated"
    assert root_entry["refresh_token"] == "refresh-rotated"
    assert not profile_auth.exists()


def test_manual_pool_refresh_keeps_identity_separate_from_singleton(
    profile_pool_env,
    monkeypatch,
):
    root_auth, profile_auth = profile_pool_env
    providers = {
        PROVIDER: {
            "tokens": {
                "access_token": "singleton-access",
                "refresh_token": "singleton-refresh",
            }
        }
    }
    entries = [
        _entry("work", access_token="work-access", refresh_token="work-refresh"),
        _entry(
            "personal",
            access_token="personal-access",
            refresh_token="personal-refresh",
        ),
    ]
    _write_store(
        root_auth,
        entries,
        providers=providers,
        suppressed_sources={PROVIDER: ["device_code"]},
    )
    pool = load_pool(PROVIDER)
    personal = next(entry for entry in pool.entries() if entry.id == "personal")
    refresh_calls: list[str] = []

    def fake_refresh(access_token, refresh_token, **_kwargs):
        refresh_calls.append(refresh_token)
        return {
            "access_token": "personal-access-rotated",
            "refresh_token": "personal-refresh-rotated",
            "last_refresh": "2026-08-20T17:00:00Z",
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake_refresh)

    refreshed = pool._refresh_entry(personal, force=True)

    assert refreshed is not None
    assert refresh_calls == ["personal-refresh"]
    persisted = {
        entry["id"]: entry
        for entry in _read_store(root_auth)["credential_pool"][PROVIDER]
    }
    assert persisted["work"]["refresh_token"] == "work-refresh"
    assert persisted["personal"]["refresh_token"] == "personal-refresh-rotated"
    assert not profile_auth.exists()


def test_global_suppression_prevents_profile_singleton_reseed(
    profile_pool_env,
):
    root_auth, profile_auth = profile_pool_env
    entries = [
        _entry("work", access_token="work-access", refresh_token="work-refresh"),
        _entry(
            "personal",
            access_token="personal-access",
            refresh_token="personal-refresh",
        ),
    ]
    _write_store(
        root_auth,
        entries,
        providers={
            PROVIDER: {
                "tokens": {
                    "access_token": "singleton-access",
                    "refresh_token": "singleton-refresh",
                }
            }
        },
        suppressed_sources={PROVIDER: ["device_code"]},
    )

    pool = load_pool(PROVIDER)

    assert [entry.id for entry in pool.entries()] == ["work", "personal"]
    assert not profile_auth.exists()


def test_two_profile_processes_share_one_single_use_refresh_post(
    profile_pool_env,
):
    root_auth, profile_auth = profile_pool_env
    _TokenHandler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TokenHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    token_url = f"http://127.0.0.1:{server.server_port}/oauth/token"
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_refresh_in_process,
            args=(
                str(profile_auth),
                str(root_auth),
                token_url,
                "global-codex",
                ready_queue,
                start_event,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert [ready_queue.get(timeout=15) for _ in processes] == [
            "ready",
            "ready",
        ]
        start_event.set()
        results = [result_queue.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    assert results == [
        ("ok", "refresh-process-rotated"),
        ("ok", "refresh-process-rotated"),
    ]
    assert _TokenHandler.calls == 1
    root_entry = _read_store(root_auth)["credential_pool"][PROVIDER][0]
    assert root_entry["refresh_token"] == "refresh-process-rotated"
    assert not profile_auth.exists()


def test_two_profiles_refreshing_different_entries_preserve_both_rotations(
    profile_pool_env,
):
    root_auth, profile_auth = profile_pool_env
    _write_store(
        root_auth,
        [
            _entry(
                "work",
                access_token="access-work-old",
                refresh_token="refresh-work-old",
            ),
            _entry(
                "personal",
                access_token="access-personal-old",
                refresh_token="refresh-personal-old",
            ),
        ],
        suppressed_sources={PROVIDER: ["device_code"]},
    )
    _TokenHandler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TokenHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    token_url = f"http://127.0.0.1:{server.server_port}/oauth/token"
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_refresh_in_process,
            args=(
                str(profile_auth),
                str(root_auth),
                token_url,
                credential_id,
                ready_queue,
                start_event,
                result_queue,
            ),
        )
        for credential_id in ("work", "personal")
    ]
    try:
        for process in processes:
            process.start()
        assert [ready_queue.get(timeout=15) for _ in processes] == [
            "ready",
            "ready",
        ]
        start_event.set()
        results = [result_queue.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    assert sorted(results) == [
        ("ok", "refresh-personal-old-rotated"),
        ("ok", "refresh-work-old-rotated"),
    ]
    assert _TokenHandler.calls == 2
    persisted = {
        entry["id"]: entry
        for entry in _read_store(root_auth)["credential_pool"][PROVIDER]
    }
    assert persisted["work"]["refresh_token"] == "refresh-work-old-rotated"
    assert persisted["personal"]["refresh_token"] == "refresh-personal-old-rotated"
    assert not profile_auth.exists()


def test_concurrent_add_status_and_remove_preserve_unrelated_owner_row(
    profile_pool_env,
):
    root_auth, profile_auth = profile_pool_env
    pool = load_pool(PROVIDER)
    original = pool.entries()[0]
    auth_mod.write_credential_pool(
        PROVIDER,
        [
            _entry(),
            _entry(
                "concurrent-add",
                access_token="access-added",
                refresh_token="refresh-added",
            ),
        ],
        target_path=root_auth,
    )

    pool._mark_exhausted(original, 429)
    after_status = {
        entry["id"]: entry
        for entry in _read_store(root_auth)["credential_pool"][PROVIDER]
    }
    assert set(after_status) == {"global-codex", "concurrent-add"}
    assert after_status["global-codex"]["last_status"] == STATUS_EXHAUSTED

    removed = pool.remove_index(1)

    assert removed is not None
    remaining = _read_store(root_auth)["credential_pool"][PROVIDER]
    assert [entry["id"] for entry in remaining] == ["concurrent-add"]
    assert not profile_auth.exists()


def test_missing_owner_entry_fails_closed_before_refresh_post(
    profile_pool_env,
    monkeypatch,
):
    root_auth, profile_auth = profile_pool_env
    pool = load_pool(PROVIDER)
    entry = pool.entries()[0]
    _write_store(root_auth, [])
    refresh_calls: list[str] = []

    def fake_refresh(access_token, refresh_token, **_kwargs):
        refresh_calls.append(refresh_token)
        return {
            "access_token": "must-not-write",
            "refresh_token": "must-not-write",
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake_refresh)

    with pytest.raises(RuntimeError, match="disappeared"):
        pool._refresh_entry(entry, force=True)

    assert refresh_calls == []
    assert _read_store(root_auth)["credential_pool"][PROVIDER] == []
    assert not profile_auth.exists()


def test_malformed_owner_fails_closed_before_refresh_post(
    profile_pool_env,
    monkeypatch,
):
    root_auth, profile_auth = profile_pool_env
    pool = load_pool(PROVIDER)
    entry = pool.entries()[0]
    root_auth.write_text("{malformed", encoding="utf-8")
    refresh_calls: list[str] = []

    def fake_refresh(access_token, refresh_token, **_kwargs):
        refresh_calls.append(refresh_token)
        return {
            "access_token": "must-not-write",
            "refresh_token": "must-not-write",
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake_refresh)

    with pytest.raises(RuntimeError, match="source changed"):
        pool._refresh_entry(entry, force=True)

    assert refresh_calls == []
    assert root_auth.read_text(encoding="utf-8") == "{malformed"
    assert not profile_auth.exists()


def test_owner_lock_timeout_fails_closed_before_refresh_post(
    profile_pool_env,
    monkeypatch,
):
    from agent import credential_pool as pool_mod

    _root_auth, profile_auth = profile_pool_env
    pool = load_pool(PROVIDER)
    entry = pool.entries()[0]
    refresh_calls: list[str] = []

    def fake_refresh(access_token, refresh_token, **_kwargs):
        refresh_calls.append(refresh_token)
        return {
            "access_token": "must-not-write",
            "refresh_token": "must-not-write",
        }

    @contextmanager
    def fail_lock(*_args, **_kwargs):
        raise TimeoutError("synthetic owner lock timeout")
        yield

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake_refresh)
    monkeypatch.setattr(pool_mod, "_auth_store_lock", fail_lock)

    with pytest.raises(TimeoutError, match="owner lock timeout"):
        pool._refresh_entry(entry, force=True)

    assert refresh_calls == []
    assert not profile_auth.exists()
