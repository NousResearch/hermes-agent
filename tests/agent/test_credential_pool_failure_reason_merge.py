import json
import time


def _write_pool(tmp_path, monkeypatch, tokens):
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    entries = [
        {"id": f"cred-{n}", "label": f"key-{n}", "auth_type": "api_key", "priority": 0,
         "source": "manual", "access_token": token}
        for n, token in enumerate(tokens, start=1)
    ]
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"openrouter": entries}}))


def _disk_row(tmp_path, cred_id):
    store = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    return next(row for row in store["credential_pool"]["openrouter"] if row["id"] == cred_id)


def test_stale_flush_keeps_a_sole_key_billing_bench(tmp_path, monkeypatch):
    _write_pool(tmp_path, monkeypatch, ["sk-or-1"])
    from agent.credential_pool import load_pool

    stale = load_pool("openrouter")
    load_pool("openrouter").mark_exhausted_and_rotate(
        status_code=403, api_key_hint="sk-or-1", failure_reason="billing")
    stale._persist()

    assert load_pool("openrouter").next_available_at() > time.time() + 30 * 60
    assert _disk_row(tmp_path, "cred-1").get("failure_reason") == "billing"


def test_stale_flush_keeps_an_unverified_billing_cooldown_short(tmp_path, monkeypatch):
    _write_pool(tmp_path, monkeypatch, ["sk-or-1", "sk-or-2"])
    from agent.credential_pool import load_pool

    stale = load_pool("openrouter")
    peer = load_pool("openrouter")
    peer.mark_exhausted_and_rotate(
        status_code=400, api_key_hint="sk-or-1", failure_reason="billing_unverified")
    peer.mark_exhausted_and_rotate(status_code=429, api_key_hint="sk-or-2", failure_reason="rate_limit")
    stale._persist()

    assert load_pool("openrouter").next_available_at() < time.time() + 5 * 60
    assert _disk_row(tmp_path, "cred-1").get("failure_reason") == "billing_unverified"
