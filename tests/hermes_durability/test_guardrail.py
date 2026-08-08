
import pytest


from hermes_durability import DurableRuntime, Guardrail
from hermes_durability.guardrail import Envelope


def ev(text: str) -> Envelope:
    return Envelope("s1", "mem", {"text": text}, "oid-1")


@pytest.fixture
def g():
    return Guardrail()


def test_clean_text_allowed(g):
    v = g.evaluate(ev("just a normal message"))
    assert v.action == "allow"


def test_api_key_redacted(g):
    v = g.evaluate(ev("my key is sk-abc123def456ghi789jkl012 ok"))
    assert v.action == "redact"
    assert "sk-abc" not in v.envelope.payload["text"]
    assert "[REDACTED:" in v.envelope.payload["text"]
    assert "ok" in v.envelope.payload["text"]


def test_private_key_blocked(g):
    v = g.evaluate(ev("-----BEGIN RSA PRIVATE KEY-----\nMIIE..."))
    assert v.action == "block"
    assert v.envelope is None


def test_ansi_glue_bypass_defeated(g):
    # secret split by ANSI styling sequences
    text = "sk-\x1b[31mabc123def456\x1b[0mghi789jkl012mno"
    v = g.evaluate(ev(text))
    assert v.action in ("redact", "block")
    assert "abc123def456" not in (v.envelope.payload["text"] if v.envelope else "")


def test_zero_width_split_defeated(g):
    text = "sk-abc​123def456ghi‍789jkl012mno"
    v = g.evaluate(ev(text))
    assert v.action == "redact"


def test_fullwidth_cjk_form_defeated(g):
    # full-width latin (NFKC folds to ascii)
    text = "ＡＫＩＡＩＯＳＦＯＤＮＮ７ＥＸＡＭＰＬＥ"
    v = g.evaluate(ev(text))
    assert v.action == "redact"


def test_cookie_header_redacted(g):
    v = g.evaluate(ev("Set-Cookie: session=8f2a9b1c7d3e4f5a; HttpOnly"))
    assert v.action == "redact"


def test_env_assignment_redacted(g):
    v = g.evaluate(ev("export STRIPE_SECRET_KEY='sk_live_abcdef123456'"))
    assert v.action == "redact"
    assert "sk_live" not in v.envelope.payload["text"]


def test_jwt_redacted(g):
    jwt = ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
           "dQw4w9WgXcQdQw4w9WgXcQ")
    v = g.evaluate(ev(f"token: {jwt}"))
    assert v.action == "redact"
    assert jwt not in v.envelope.payload["text"]


def test_hot_reload_policies(g):
    g.load_policy_dicts([{"id": "custom.word", "pattern": r"\bxyzzy\b",
                          "action": "drop"}])
    assert g.evaluate(ev("say xyzzy")).action == "drop"
    # old rules replaced
    assert g.evaluate(ev("sk-abc123def456ghi789jkl012")).action == "allow"


def test_audit_written_before_send(tmp_path):
    db = str(tmp_path / "a.db")

    class NeverSend:
        def send(self, envelope, idempotency_key):
            raise AssertionError("must not be called for blocked envelope")

    rt = DurableRuntime(db, adapters={"mem": NeverSend()}, start_worker=False)
    with rt.transaction("s1") as txn:
        txn.enqueue_outbound("mem", {"text": "-----BEGIN PRIVATE KEY-----"})
    rt.worker.drain_once()
    rows = rt.journal._conn.execute(
        "SELECT action, policy_id FROM audit_log").fetchall()
    assert ("block", "secret.private-key-block") in rows
    # blocked row is terminal, never retried
    status = rt.journal._conn.execute(
        "SELECT status FROM outbox").fetchone()[0]
    assert status == "blocked"
    rt.close()


def test_guardrail_runs_on_delivery_path(tmp_path):
    db = str(tmp_path / "b.db")
    sent = {}

    class Recv:
        def send(self, envelope, idempotency_key):
            sent[idempotency_key] = envelope.payload["text"]
            return {}

    rt = DurableRuntime(db, adapters={"mem": Recv()}, start_worker=False)
    with rt.transaction("s1") as txn:
        oid = txn.enqueue_outbound(
            "mem", {"text": "key=sk-abc123def456ghi789jkl012"})
    rt.worker.drain_once()
    assert "sk-abc" not in sent[oid]
    rt.close()
