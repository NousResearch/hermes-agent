import json
import threading

import pytest

from plugins.memory.obsidimem import ObsidimemProvider

_DEFAULT_CONFIG = {
    "api_base_url": "http://127.0.0.1:8000",
    "observer_name": "hermes",
    "observed_name": "doug",
    "recall_mode": "hybrid",
    "budget": 1200,
    "timeout": 60.0,
    "trigger_dreamer_on_session_end": False,
}


class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self._data = data or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class FakeClient:
    """Records all HTTP calls; returns sensible defaults."""

    def __init__(self, **kwargs):
        self.calls: list[tuple[str, str, object]] = []

    def post(self, url, *, json=None, **kwargs):
        self.calls.append(("POST", url, json))
        if "/memory/sessions" in url:
            return FakeResponse({"id": "obs-session-abc"})
        return FakeResponse({})

    def get(self, url, *, params=None, **kwargs):
        self.calls.append(("GET", url, params))
        return FakeResponse({})

    def patch(self, url, **kwargs):
        self.calls.append(("PATCH", url, None))
        return FakeResponse({})

    def close(self):
        self.calls.append(("CLOSE", None, None))


@pytest.fixture
def fake_client(monkeypatch):
    """Patch httpx.Client so the plugin gets FakeClient on initialize()."""
    client = FakeClient()
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    return client


@pytest.fixture
def provider(fake_client, tmp_path):
    """Initialized ObsidimemProvider backed by FakeClient."""
    (tmp_path / "obsidimem.json").write_text(json.dumps(_DEFAULT_CONFIG))
    p = ObsidimemProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    return p


# ---------------------------------------------------------------------------
# on_session_switch
# ---------------------------------------------------------------------------

def test_session_switch_ends_old_session(provider, fake_client):
    """PATCH /memory/sessions/<old_id>/end must fire on switch."""
    old_id = provider._session_id  # "obs-session-abc" from fixture
    fake_client.calls.clear()

    provider.on_session_switch("session-2", parent_session_id="session-1")

    patch_calls = [(m, u) for m, u, _ in fake_client.calls if m == "PATCH"]
    assert any(f"/memory/sessions/{old_id}/end" in u for _, u in patch_calls)


def test_session_switch_creates_new_obsidimem_session(provider, fake_client):
    """POST /memory/sessions must fire to mint a new session under new_session_id."""
    fake_client.calls.clear()

    provider.on_session_switch("session-2", parent_session_id="session-1")

    post_calls = [(m, u, b) for m, u, b in fake_client.calls if m == "POST"]
    session_posts = [(u, b) for _, u, b in post_calls if "/memory/sessions" in u]
    assert len(session_posts) == 1
    assert session_posts[0][1]["metadata"]["hermes_session_id"] == "session-2"


def test_session_switch_updates_cached_session_id(provider, fake_client):
    """`_session_id` must reflect the new obsidimem session after switch."""
    provider.on_session_switch("session-2", parent_session_id="session-1")

    assert provider._session_id == "obs-session-abc"  # FakeClient always returns this
    assert provider._session_initialized is True


def test_session_switch_clears_prefetch_state(provider, fake_client):
    """Stale prefetch result must be cleared so turn 1 of new session isn't polluted."""
    with provider._prefetch_lock:
        provider._prefetch_result = "stale context from old session"

    provider.on_session_switch("session-2", parent_session_id="session-1")

    with provider._prefetch_lock:
        assert provider._prefetch_result == ""


def test_session_switch_triggers_dreamer_on_reset(monkeypatch, tmp_path, fake_client):
    """Dreamer must fire on reset=True when trigger_dreamer_on_session_end is set."""
    cfg = {**_DEFAULT_CONFIG, "trigger_dreamer_on_session_end": True}
    (tmp_path / "obsidimem.json").write_text(json.dumps(cfg))
    p = ObsidimemProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    fake_client.calls.clear()

    p.on_session_switch("session-2", parent_session_id="session-1", reset=True)

    dreamer_posts = [u for m, u, _ in fake_client.calls if m == "POST" and "/dreamer" in u]
    assert len(dreamer_posts) == 1


def test_session_switch_does_not_trigger_dreamer_on_resume(monkeypatch, tmp_path, fake_client):
    """Dreamer must NOT fire on resume/branch (reset=False), even if configured."""
    cfg = {**_DEFAULT_CONFIG, "trigger_dreamer_on_session_end": True}
    (tmp_path / "obsidimem.json").write_text(json.dumps(cfg))
    p = ObsidimemProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    fake_client.calls.clear()

    p.on_session_switch("session-2", parent_session_id="session-1", reset=False)

    dreamer_posts = [u for m, u, _ in fake_client.calls if m == "POST" and "/dreamer" in u]
    assert dreamer_posts == []


def test_session_switch_empty_id_is_noop(provider, fake_client):
    """Empty or None new_session_id must not mutate provider state."""
    old_id = provider._session_id
    fake_client.calls.clear()

    provider.on_session_switch("")
    provider.on_session_switch(None)  # type: ignore[arg-type]

    assert provider._session_id == old_id
    assert fake_client.calls == []


def test_session_switch_cron_skipped_is_noop(monkeypatch, tmp_path, fake_client):
    """Cron-skipped providers must ignore the switch entirely."""
    (tmp_path / "obsidimem.json").write_text(json.dumps(_DEFAULT_CONFIG))
    p = ObsidimemProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cron")
    fake_client.calls.clear()

    p.on_session_switch("session-2")

    assert fake_client.calls == []


# ---------------------------------------------------------------------------
# on_memory_write
# ---------------------------------------------------------------------------

def test_memory_write_add_posts_observation(provider, fake_client):
    """add action must POST an explicit-level observation to obsidimem."""
    fake_client.calls.clear()

    provider.on_memory_write("add", "memory", "Doug prefers terse responses")
    provider._write_thread.join(timeout=2.0)

    obs_posts = [b for m, u, b in fake_client.calls if m == "POST" and "/memory/observations" in u]
    assert len(obs_posts) == 1
    obs = obs_posts[0]["observations"][0]
    assert obs["level"] == "explicit"
    assert "Doug prefers terse responses" in obs["content"]
    assert obs["observer_name"] == "hermes"
    assert obs["observed_name"] == "doug"


def test_memory_write_replace_posts_observation(provider, fake_client):
    """replace action must also POST to obsidimem (same as add)."""
    fake_client.calls.clear()

    provider.on_memory_write("replace", "user", "Doug is a networking veteran")
    provider._write_thread.join(timeout=2.0)

    obs_posts = [b for m, u, b in fake_client.calls if m == "POST" and "/memory/observations" in u]
    assert len(obs_posts) == 1


def test_memory_write_remove_is_noop(provider, fake_client):
    """remove action must not call any API (no observation delete endpoint)."""
    fake_client.calls.clear()

    provider.on_memory_write("remove", "memory", "some old fact")

    obs_posts = [u for m, u, _ in fake_client.calls if m == "POST" and "/memory/observations" in u]
    assert obs_posts == []


def test_memory_write_is_non_blocking(provider, fake_client):
    """on_memory_write must return immediately (background thread)."""
    import time

    slow_done = threading.Event()
    original_post = fake_client.post

    def slow_post(url, *, json=None, **kwargs):
        if "/memory/observations" in url:
            time.sleep(0.3)
            slow_done.set()
        return original_post(url, json=json, **kwargs)

    fake_client.post = slow_post

    start = time.monotonic()
    provider.on_memory_write("add", "memory", "test content")
    elapsed = time.monotonic() - start

    assert elapsed < 0.1, f"on_memory_write blocked for {elapsed:.2f}s"
    provider._write_thread.join(timeout=2.0)
    assert slow_done.is_set()


def test_memory_write_cron_skipped_is_noop(monkeypatch, tmp_path, fake_client):
    """Cron-skipped providers must not post observations."""
    (tmp_path / "obsidimem.json").write_text(json.dumps(_DEFAULT_CONFIG))
    p = ObsidimemProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cron")
    fake_client.calls.clear()

    p.on_memory_write("add", "memory", "should not be stored")

    assert p._write_thread is None
    obs_posts = [u for m, u, _ in fake_client.calls if m == "POST" and "/memory/observations" in u]
    assert obs_posts == []
