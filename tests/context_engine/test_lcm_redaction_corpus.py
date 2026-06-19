from __future__ import annotations

import json
import sqlite3

from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.dag import SummaryDAG, SummaryNode
from plugins.context_engine.lcm.store import MessageStore


def _redaction_config(db_path, hermes_home) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        sensitive_patterns_enabled=True,
        sensitive_patterns=["all"],
        encryption_enabled=False,
    )


def _joined(*parts: str) -> str:
    return "".join(parts)


def _private_key_block() -> str:
    header = _joined("-" * 5, "BEGIN ", "OPENSSH ", "PRIVATE ", "KEY", "-" * 5)
    footer = _joined("-" * 5, "END ", "OPENSSH ", "PRIVATE ", "KEY", "-" * 5)
    return "\n".join([header, _joined("b3Bl", "bnNzaC1r", "ZXktdjE", "AAAA"), footer])


def _secret_corpus() -> dict[str, str]:
    jwt = ".".join(
        [
            _joined("eyJ", "hbG", "ciO", "iJI", "UzI", "1Ni", "J9"),
            _joined("eyJ", "zdW", "IiO", "iJs", "Y20", "tdG", "Vz", "dCJ", "9"),
            _joined("c2l", "nbm", "F0d", "XJl", "X3N", "lbn", "Rpb", "mVs"),
        ]
    )
    return {
        "onepassword_ref": _joined("op", "://", "Engineering/LCM Sentinel/password"),
        "bearer_jwt": _joined("Authorization: ", "Bear", "er ", jwt),
        "cookie": _joined("Cookie: sessionid=", "a" * 32, "; csrftoken=", "b" * 32),
        "aws_access_key": _joined("AK", "IA", "ABCDEFGHIJKLMNOP"),
        "github_token": _joined("gh", "p_", "c" * 36),
        "openai_key": _joined("sk", "-", "proj", "-", "d" * 48),
        "private_key": _private_key_block(),
        "homelab_password": _joined("universal homelab password: ", "CorrectHorseBattery99!"),
    }


def _scan_sqlite_text(db_path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        chunks: list[str] = []
        table_names = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        for table in table_names:
            quoted = '"' + table.replace('"', '""') + '"'
            try:
                rows = conn.execute(f"SELECT * FROM {quoted}").fetchall()
            except sqlite3.DatabaseError:
                continue
            for row in rows:
                for value in row:
                    if isinstance(value, bytes):
                        chunks.append(value.decode("utf-8", errors="ignore"))
                    elif value is not None:
                        chunks.append(str(value))
        chunks.append("\n".join(conn.iterdump()))
        return "\n".join(chunks)
    finally:
        conn.close()


def test_redaction_corpus_is_removed_from_raw_dag_summary_and_fts(tmp_path):
    db_path = tmp_path / "profile" / "lcm.db"
    cfg = _redaction_config(db_path, tmp_path / "profile")
    secrets = _secret_corpus()
    plaintext = "\n".join(f"{name}: {secret}" for name, secret in secrets.items())

    store = MessageStore(db_path, ingest_protection_config=cfg, hermes_home=str(tmp_path / "profile"))
    dag = SummaryDAG(db_path, ingest_protection_config=cfg)

    store_id = store.append(
        "session-redaction",
        {
            "role": "user",
            "content": plaintext,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "leaky_tool",
                        "arguments": json.dumps(
                            {
                                "authorization": secrets["bearer_jwt"],
                                "cookie": secrets["cookie"],
                                "api_key": secrets["openai_key"],
                            }
                        ),
                    },
                }
            ],
        },
        token_estimate=100,
    )
    dag.add_node(
        SummaryNode(
            session_id="session-redaction",
            depth=0,
            summary="summary repeated corpus\n" + plaintext,
            token_count=50,
            source_token_count=100,
            source_ids=[store_id],
            source_type="messages",
            created_at=123.0,
            expand_hint="expand repeated " + secrets["openai_key"],
        )
    )

    stored = store.get(store_id)
    assert stored is not None
    assert "LCM sensitive redaction" in stored["content"]
    assert "LCM sensitive redaction" in json.dumps(stored["tool_calls"], ensure_ascii=False)

    node = dag.get_session_nodes("session-redaction")[0]
    assert "LCM sensitive redaction" in node.summary
    assert "LCM sensitive redaction" in node.expand_hint

    sqlite_text = _scan_sqlite_text(db_path)
    for name, secret in secrets.items():
        assert secret not in stored["content"], name
        assert secret not in json.dumps(stored["tool_calls"], ensure_ascii=False), name
        assert secret not in node.summary, name
        assert secret not in node.expand_hint, name
        assert secret not in sqlite_text, name

    store.close()
    dag.close()
