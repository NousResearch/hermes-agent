import json
import os

import pytest

from hermes_cli.session_presence import (
    clear_session_presence,
    list_session_presence,
    write_session_presence,
)


@pytest.fixture(autouse=True)
def isolate_presence_env(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_PRESENCE_DIR", raising=False)


def test_session_presence_write_list_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "host-a")
    monkeypatch.setattr(os, "getpid", lambda: 1234)

    record = write_session_presence(
        session_id="sid-1",
        session_key="stored-1",
        status="working",
        title="Build feature",
        model="qwen",
        cwd="/repo",
        source="tui_gateway",
        client="desktop",
        profile="default",
        endpoint="ws://127.0.0.1:8765",
        hermes_home=tmp_path,
        now=100.0,
    )

    assert record["host"] == "host-a"
    assert record["pid"] == 1234
    assert record["expires_at"] > record["updated_at"]

    rows = list_session_presence(hermes_home=tmp_path, now=101.0)
    assert [row["session_id"] for row in rows] == ["sid-1"]
    assert rows[0]["session_key"] == "stored-1"
    assert rows[0]["status"] == "working"

    assert clear_session_presence(session_id="sid-1", hermes_home=tmp_path) == 1
    assert list_session_presence(hermes_home=tmp_path, now=101.0) == []


def test_session_presence_filters_expired_records(tmp_path):
    write_session_presence(
        session_id="old",
        ttl_seconds=5,
        hermes_home=tmp_path,
        instance_id="instance",
        now=10.0,
    )

    assert list_session_presence(hermes_home=tmp_path, now=20.0) == []
    assert list_session_presence(hermes_home=tmp_path, now=20.0, include_expired=True)[0][
        "session_id"
    ] == "old"


def test_session_presence_dedupes_duplicate_runtime_records(tmp_path):
    older = write_session_presence(
        session_id="sid-1",
        session_key="old-key",
        client="hphone",
        profile="taro",
        hermes_home=tmp_path,
        instance_id="instance",
        now=10.0,
    )
    root = tmp_path / "session-presence" / "active"
    conflict = dict(older)
    conflict["session_key"] = "new-key"
    conflict["updated_at"] = 12.0
    conflict["expires_at"] = 102.0
    (root / "instance.sid-1.sync-conflict.json").write_text(
        json.dumps(conflict),
        encoding="utf-8",
    )

    rows = list_session_presence(hermes_home=tmp_path, now=13.0)

    assert [row["session_id"] for row in rows] == ["sid-1"]
    assert rows[0]["session_key"] == "new-key"


def test_session_presence_dedupes_endpoint_successors(tmp_path):
    write_session_presence(
        session_id="old-runtime",
        session_key="old-key",
        client="hphone",
        profile="taro",
        endpoint="tmux://taro/hermes-phone",
        hermes_home=tmp_path,
        instance_id="old-instance",
        now=10.0,
    )
    write_session_presence(
        session_id="new-runtime",
        session_key="new-key",
        client="hphone",
        profile="taro",
        endpoint="tmux://taro/hermes-phone",
        hermes_home=tmp_path,
        instance_id="new-instance",
        now=12.0,
    )

    rows = list_session_presence(hermes_home=tmp_path, now=13.0)

    assert [row["session_id"] for row in rows] == ["new-runtime"]
    assert rows[0]["session_key"] == "new-key"


def test_session_presence_dir_env_overrides_hermes_home(tmp_path, monkeypatch):
    presence_dir = tmp_path / "shared-presence"
    monkeypatch.setenv("HERMES_SESSION_PRESENCE_DIR", str(presence_dir))

    write_session_presence(session_id="shared", hermes_home=tmp_path / "ignored", now=10.0)

    assert list_session_presence(hermes_home=tmp_path / "ignored", now=11.0)[0][
        "session_id"
    ] == "shared"
    assert not (tmp_path / "ignored" / "session-presence" / "active").exists()
