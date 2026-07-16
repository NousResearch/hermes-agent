from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import hermes_state
from hermes_state import SessionDB


def run_case(case: dict):
    kind = case["kind"]
    if kind == "placeholders":
        helper = _helper("_sql_placeholders")
        return {
            "return": [helper(values) for values in case["values"]],
            "messages": [],
            "db": [],
        }
    if kind == "denorm_flag":
        return _run_denorm_case(case)
    if kind == "title_search":
        return _run_title_search_case(case)
    raise AssertionError(f"unknown state_ext case kind: {kind!r}")


def _run_denorm_case(case: dict):
    helper = _helper("_session_list_denorm_enabled")
    out = []
    with _patched_env(case.get("env") or {}):
        if "config_error" in case:
            import hermes_cli.config as config

            old_read_raw_config = config.read_raw_config
            try:
                config.read_raw_config = lambda: (_ for _ in ()).throw(
                    RuntimeError(case["config_error"])
                )
                for enabled in case["configs"]:
                    _write_dashboard_flag(bool(enabled))
                    out.append(helper())
            finally:
                config.read_raw_config = old_read_raw_config
        else:
            for enabled in case["configs"]:
                _write_dashboard_flag(bool(enabled))
                out.append(helper())
    return {"return": out, "messages": [], "db": []}


def _run_title_search_case(case: dict):
    db_path = Path(os.environ["HERMES_HOME"]) / (case["name"].replace(" ", "_") + ".db")
    db = SessionDB(db_path=db_path)
    try:
        _seed_title_search_fixture(db)
        rows = db.search_sessions_by_title(
            case["query"],
            limit=case["limit"],
            include_archived=case["include_archived"],
        )
        selected = [
            {
                "id": row["id"],
                "title": row.get("title"),
                "display_name": row.get("display_name"),
                "source": row.get("source"),
                "preview": row.get("preview"),
            }
            for row in rows
        ]
        persisted = [
            dict(row)
            for row in db._conn.execute(
                "SELECT id, source, title, display_name FROM sessions ORDER BY id"
            ).fetchall()
        ]
        return {"return": selected, "messages": [], "db": persisted}
    finally:
        db.close()


def _seed_title_search_fixture(db: SessionDB) -> None:
    db.create_session(session_id="dc", source="discord")
    db.record_gateway_session_peer(
        "dc",
        source="discord",
        session_key="agent:main:discord:thread:dc",
        chat_id="123",
        display_name="Daemonarchy / #general / Deploy Notes",
    )
    db.append_message("dc", role="user", content="discord preview")

    db.create_session(session_id="tg", source="telegram")
    db.set_session_title("tg", "General chatter")
    db.append_message("tg", role="user", content="telegram preview")

    db.create_session(session_id="titled", source="cli")
    db.set_session_title("titled", "Deploy pipeline")
    db.append_message("titled", role="user", content="cli preview")

    db.create_session(session_id="wild", source="cli")
    db.set_session_title("wild", "100% coverage plan")
    db.append_message("wild", role="user", content="wildcard preview")

    db.create_session(session_id="plain", source="cli")
    db.set_session_title("plain", "No platform words here")

    db.create_session(session_id="untitled-discord", source="discord")


def _helper(name: str):
    try:
        import hermes_state_ext
    except ModuleNotFoundError:
        return getattr(hermes_state, name)
    return getattr(hermes_state_ext, name)


def _write_dashboard_flag(enabled: bool) -> None:
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config_path.write_text(
        "dashboard:\n"
        f"  session_list_denorm: {json.dumps(enabled)}\n",
        encoding="utf-8",
    )


@contextmanager
def _patched_env(values: dict[str, str]):
    tracked = set(values) | {"HERMES_SESSION_LIST_DENORM"}
    old = {key: os.environ.get(key) for key in tracked}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        if "HERMES_SESSION_LIST_DENORM" not in values:
            os.environ.pop("HERMES_SESSION_LIST_DENORM", None)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
