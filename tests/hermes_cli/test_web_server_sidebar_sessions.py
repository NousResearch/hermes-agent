"""Regression tests for the batched sidebar endpoint's profile scoping.

``/api/profiles/sessions/sidebar`` returns three windows per refresh: recents,
cron, and messaging-platform conversations. Recents was scoped to
``recents_profile`` from the start, but messaging was unconditionally
cross-profile — and every messaging read shares one bounded ``messaging_limit``
window. So a profile with many conversations filled the window from the union
and crowded the quieter profiles out of it, leaving their WeChat / Telegram
sections looking truncated (or empty) no matter which profile was selected.

These tests pin ``messaging_profile``: a concrete profile windows only its own
rows, ``all`` keeps the unified view, and omitting the param keeps the old
cross-profile default so an older desktop against a newer backend is unaffected.
"""
import pytest


_MESSAGING_EXCLUDE = "cron,cli,codex,desktop,gateway,local,tui"


@pytest.fixture
def isolated_profiles(monkeypatch, _isolate_hermes_home):
    """Isolated default home + one named profile, each with its own state.db."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker": worker_home}


@pytest.fixture
def client(monkeypatch, isolated_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _seed_conversation(db_path, session_id, source="weixin"):
    """One messaging conversation with a message, so min_messages=1 keeps it."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id=session_id, source=source)
        db.append_message(session_id, role="user", content="hi")
    finally:
        db.close()


def _sidebar_query(**extra):
    params = {
        "recents_profile": "worker",
        "recents_limit": "20",
        # Mirrors the desktop's taxonomy: recents is local chats only, so cron
        # and every messaging source are excluded from it.
        "recents_exclude": "cron,weixin",
        "cron_limit": "50",
        "messaging_limit": "100",
        "messaging_exclude": _MESSAGING_EXCLUDE,
        **extra,
    }
    return "/api/profiles/sessions/sidebar?" + "&".join(
        f"{k}={v}" for k, v in params.items()
    )


def _messaging_ids(response):
    assert response.status_code == 200
    return {s["id"] for s in response.json()["messaging"]["sessions"]}


def test_messaging_slice_windows_only_the_requested_profile(client, isolated_profiles):
    """A concrete ``messaging_profile`` returns that profile's rows alone."""
    _seed_conversation(isolated_profiles["default"] / "state.db", "wx-default")
    _seed_conversation(isolated_profiles["worker"] / "state.db", "wx-worker")

    assert _messaging_ids(client.get(_sidebar_query(messaging_profile="worker"))) == {
        "wx-worker"
    }
    assert _messaging_ids(client.get(_sidebar_query(messaging_profile="default"))) == {
        "wx-default"
    }


def test_messaging_slice_stays_unified_for_all_and_when_omitted(
    client, isolated_profiles
):
    """``all`` keeps the union, and so does omitting the param entirely —
    back-compat for an older desktop that sends no ``messaging_profile``."""
    _seed_conversation(isolated_profiles["default"] / "state.db", "wx-default")
    _seed_conversation(isolated_profiles["worker"] / "state.db", "wx-worker")

    both = {"wx-default", "wx-worker"}
    assert both <= _messaging_ids(client.get(_sidebar_query(messaging_profile="all")))
    assert both <= _messaging_ids(client.get(_sidebar_query()))


def test_messaging_total_counts_only_the_scoped_rows(client, isolated_profiles):
    """``messaging.total`` is what the desktop resolves a platform section's
    exact count from, so it has to narrow with the scope — otherwise a scoped
    section advertises the union's count and offers a "load more" that can
    never deliver."""
    _seed_conversation(isolated_profiles["default"] / "state.db", "wx-default-1")
    _seed_conversation(isolated_profiles["default"] / "state.db", "wx-default-2")
    _seed_conversation(isolated_profiles["worker"] / "state.db", "wx-worker")

    scoped = client.get(_sidebar_query(messaging_profile="worker"))
    assert scoped.status_code == 200
    assert scoped.json()["messaging"]["total"] == 1

    unified = client.get(_sidebar_query(messaging_profile="all"))
    assert unified.status_code == 200
    assert unified.json()["messaging"]["total"] == 3


def test_messaging_scope_does_not_narrow_the_other_slices(client, isolated_profiles):
    """Only messaging moves: cron stays cross-profile and recents keeps
    following ``recents_profile``, so scoping one slice can't silently
    re-window the others."""
    _seed_conversation(isolated_profiles["default"] / "state.db", "cron-default", source="cron")
    _seed_conversation(isolated_profiles["worker"] / "state.db", "cron-worker", source="cron")
    _seed_conversation(isolated_profiles["worker"] / "state.db", "local-worker", source="desktop")

    response = client.get(_sidebar_query(messaging_profile="worker"))
    assert response.status_code == 200
    data = response.json()

    # cron is deliberately unscoped by messaging_profile.
    assert {s["id"] for s in data["cron"]["sessions"]} == {"cron-default", "cron-worker"}
    # recents still honors recents_profile=worker.
    assert {s["id"] for s in data["recents"]["sessions"]} == {"local-worker"}
