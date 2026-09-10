"""Persisted transcript -> REST: creation evidence, not PR mentions."""
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.web_routers import profiles
from hermes_state import SessionDB

URL = "https://github.com/example/base/pull/7"
REPLACEMENT = "https://github.com/example/base/pull/8"


def _terminal(output=URL, exit_code=0, **extra):
    return {"output": output, "exit_code": exit_code, **extra}


def _persist(db, session, name, args, result, *, linked=True):
    call_id = f"call-{db.get_active_message_watermark(session)}"
    db.append_message(session, "assistant", tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }])
    content = json.dumps(result)
    if session == "mixed":
        content = content.replace("/", "\\/")
    db.append_message(session, "tool", tool_name=name,
                      tool_call_id=call_id if linked else "missing-call", content=content)


def _client(monkeypatch, tmp_path, targets):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **k: targets)
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    return TestClient(app)


@pytest.mark.parametrize("compacted", [False, True], ids=["live", "compacted"])
def test_only_successful_linked_creations_survive_later_research(tmp_path, monkeypatch, compacted):
    nested_code = 'from hermes_tools import terminal\nprint(terminal(command="gh pr create --fill"))'
    nested_result = {"status": "success", "exit_code": 0, "tool_calls_made": 1,
                     "output": repr(_terminal("body scrub done\n" + URL)) + "\n"}
    cases = {
        "bare": ("terminal", {"command": "gh pr create --fill"}, _terminal(), True),
        "mixed": ("terminal", {"command": "git push 2>&1 | tail -1; echo 'body scrub done'; gh pr create --fill"}, _terminal("body scrub done\n" + URL + "\n"), True),
        "create-heredoc": ("terminal", {"command": "git push && gh pr create --body-file - <<'EOF'\n# Body\nReference gh pr view\nEOF"}, _terminal("pushed\n" + URL), True),
        "create-tail": ("terminal", {"command": "git push; gh pr create --fill 2>&1 | tail -2"}, _terminal("body scrub done\n" + URL), True),
        "create-redirect": ("terminal", {"command": "gh pr create --fill 2>&1"}, _terminal(), True),
        "tail-file": ("terminal", {"command": "gh pr create --fill | tail -2 saved.txt"}, _terminal(), False),
        "tail-skipped": ("terminal", {"command": "true || gh pr create --fill | tail -2"}, _terminal(), False),
        "heredoc-followed": ("terminal", {"command": "cat <<'EOF'\nbody\nEOF\ngh pr view"}, _terminal(), False),
        "nested": ("execute_code", {"code": nested_code}, nested_result, True),
        "view": ("terminal", {"command": "gh pr view --json url --jq .url"}, _terminal(), False),
        "review": ("terminal", {"command": "gh pr review --approve"}, _terminal(), False),
        "comment": ("terminal", {"command": "gh pr view # gh pr create"}, _terminal(), False),
        "body": ("terminal", {"command": "gh pr comment 7 --body 'notes; gh pr create'"}, _terminal(), False),
        "heredoc": ("terminal", {"command": "cat <<'EOF'\ngh pr create\nEOF"}, _terminal(), False),
        "prose": ("terminal", {"command": "gh pr create --fill"}, _terminal("Review " + URL), False),
        "failed": ("terminal", {"command": "gh pr create --fill"}, _terminal(exit_code=1), False),
        "error": ("terminal", {"command": "gh pr create --fill"}, _terminal(error="failed"), False),
        "unknown-status": ("terminal", {"command": "gh pr create --fill"}, {"output": URL}, False),
        "masked-failure": ("terminal", {"command": "gh pr create || true"}, _terminal(), False),
        "skipped": ("terminal", {"command": "true || gh pr create"}, _terminal(), False),
        "skipped-newline": ("terminal", {"command": "true ||\n# still conditional\ngh pr create"}, _terminal(), False),
        "background-newline": ("terminal", {"command": "gh pr create &\n"}, _terminal(), False),
        "unlinked": ("terminal", {"command": "gh pr create --fill"}, _terminal(), False),
        "bad-host": ("terminal", {"command": "gh pr create --fill"}, _terminal(URL.replace("github.com", "github.com.evil")), False),
        "bad-number": ("terminal", {"command": "gh pr create --fill"}, _terminal(URL[:-1] + "0"), False),
        "bad-owner": ("terminal", {"command": "gh pr create --fill"}, _terminal(URL.replace("/example/", "/../")), False),
        "ambiguous": ("terminal", {"command": "gh pr create --fill"}, _terminal(URL + "\n" + REPLACEMENT), False),
        "nested-failed": ("execute_code", {"code": nested_code}, {**nested_result, "output": json.dumps(_terminal(exit_code=1))}, False),
        "nested-mention": ("execute_code", {"code": '# gh pr create\nprint("reference")'}, nested_result, False),
        "nested-body": ("execute_code", {"code": 'from hermes_tools import terminal\nprint(terminal(command="gh pr comment 7 --body \'gh pr create\'"))'}, nested_result, False),
        "nested-fake": ("execute_code", {"code": 'from hermes_tools import terminal\nif False:\n terminal(command="gh pr create")\nprint({"output": "reference", "exit_code": 0})'}, nested_result, False),
    }
    with SessionDB(tmp_path / "state.db") as db:
        for session, (name, args, result, created) in cases.items():
            db.create_session(session_id=session, source="desktop")
            _persist(db, session, name, args, result, linked=session != "unlinked")
            if compacted:
                db.archive_and_compact(session, [{"role": "user", "content": "summary"}])
    with _client(monkeypatch, tmp_path, [("worker", tmp_path)]) as client:
        response = client.post("/api/profiles/sessions/pull-requests", json={"ids": list(cases)})
        assert response.status_code == 200
        assert response.json()["pull_requests"] == {
            session: {"number": 7, "url": URL}
            for session, (_, _, _, created) in cases.items() if created
        }
        with SessionDB(tmp_path / "state.db") as db:
            for session, (_, _, _, created) in cases.items():
                if created:
                    _persist(db, session, "terminal", {"command": "gh pr create --fill"}, _terminal(REPLACEMENT))
                    _persist(db, session, "terminal", {"command": "gh pr view"}, _terminal(URL))
        response = client.post("/api/profiles/sessions/pull-requests", json={"ids": list(cases)})
    assert response.status_code == 200
    assert response.json()["pull_requests"] == {
        session: {"number": 8, "url": REPLACEMENT}
        for session, (_, _, _, created) in cases.items() if created
    }
    assert set(response.json()["scanned"]) == set(cases)


@pytest.mark.parametrize("inventory_failure", [False, True])
def test_scan_uses_real_profile_discovery(tmp_path, monkeypatch, inventory_failure):
    from hermes_cli import profiles as profiles_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        db.create_session("discovered", "desktop")
        _persist(db, "discovered", "terminal", {"command": "gh pr create --fill"}, _terminal())
    if inventory_failure:
        def unavailable():
            raise OSError("inventory unavailable")
        monkeypatch.setattr(profiles_mod, "list_profiles", unavailable)
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    with TestClient(app) as client:
        response = client.post("/api/profiles/sessions/pull-requests", json={"ids": ["discovered"]})
    assert response.status_code == 200
    assert response.json()["pull_requests"] == {"discovered": {"number": 7, "url": URL}}
    assert response.json()["scanned"] == ([] if inventory_failure else ["discovered"])


@pytest.mark.parametrize("unavailable", ["corrupt", "missing", "inventory"])
def test_unread_profiles_remain_retryable_until_real_read_succeeds(tmp_path, monkeypatch, unavailable):
    healthy, unread = tmp_path / "healthy", tmp_path / "unread"
    healthy.mkdir()
    unread.mkdir()
    with SessionDB(healthy / "state.db") as db:
        db.create_session(session_id="known-empty", source="desktop")
    if unavailable == "corrupt":
        (unread / "state.db").write_bytes(b"not a SQLite database")
    targets = [("healthy", healthy), ("unread", unread)]
    with _client(monkeypatch, tmp_path, targets) as client:
        if unavailable == "inventory":
            def failed_inventory(*args, errors=None, **kwargs):
                errors.append({"error": "profile-inventory-unavailable"}) if errors is not None else None
                return [("healthy", healthy)]
            monkeypatch.setattr(profiles, "_profile_targets", failed_inventory)
        payload = {"ids": ["known-empty", "unread-session"]}
        first = client.post("/api/profiles/sessions/pull-requests", json=payload)
        assert first.status_code == 200
        assert "unread-session" not in first.json()["scanned"]
        assert "unread-session" not in first.json()["pull_requests"]
        if unavailable == "corrupt":
            (unread / "state.db").unlink()
        with SessionDB(unread / "state.db") as db:
            db.create_session(session_id="unread-session", source="desktop")
            _persist(db, "unread-session", "terminal", {"command": "gh pr create --fill"}, _terminal())
        monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **k: targets)
        retried = client.post("/api/profiles/sessions/pull-requests", json=payload)
        assert retried.status_code == 200
        assert retried.json()["pull_requests"] == {"unread-session": {"number": 7, "url": URL}}
        assert set(retried.json()["scanned"]) == set(payload["ids"])
