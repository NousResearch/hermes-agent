"""An entitlement bench on one model must not take a sole Codex credential away from aux tasks on OTHER models.

Real incident: one ChatGPT account (a single ``openai-codex`` pool entry shared by 8 profiles). An ad-hoc
``hermes -m gpt-5.5`` drew the ChatGPT-account entitlement 400, which benches (credential, gpt-5.5) for a year.
Main-agent calls name their model and kept working, but auxiliary resolution selected WITHOUT a model, and a
model-less select honours every active model cooldown -> "openai-codex requested but no Codex OAuth token found"
and "No auxiliary LLM provider for compression" in every profile. ``hermes auth reset`` cleared it, but a
long-lived process holding the bench in memory could max-merge it back on its next flush.

Real ``agent.credential_pool`` against a real temp auth store, production bench path
(``mark_exhausted_and_rotate(..., failure_reason="model_entitlement")``).
"""
import json
import time

import pytest

BENCHED = "gpt-5.5"
AUX_MODEL = "gpt-6-sol"
TOKEN = "tok-sole-account"


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    for var in ("OPENAI_API_KEY", "HERMES_CODEX_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": "sole", "label": "acct", "auth_type": "oauth", "priority": 0, "source": "manual",
         "access_token": TOKEN, "refresh_token": "rt", "expires_at_ms": 4_000_000_000_000}]}}))
    return root


def _bench(pool):
    pool.mark_exhausted_and_rotate(status_code=400, api_key_hint=TOKEN, credential_id="sole",
                                   failure_reason="model_entitlement", model=BENCHED)


def _disk_cooldowns(root):
    row = json.loads((root / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    return {m: u for m, u in (row.get("model_cooldowns") or {}).items() if u > time.time()}


def test_aux_task_on_another_model_still_resolves_the_sole_credential(root, monkeypatch):
    from agent import auxiliary_client as ac
    from agent.credential_pool import load_pool

    monkeypatch.setattr(ac, "_read_codex_singleton_token", lambda: None)   # pool is the only authority
    _bench(load_pool("openai-codex"))
    assert set(_disk_cooldowns(root)) == {BENCHED}

    token, _base = ac._resolve_codex_credential_and_base(AUX_MODEL)
    assert token == TOKEN                                  # unrelated model: credential is usable
    assert ac._resolve_codex_credential_and_base(BENCHED)[0] is None    # the benched model stays benched
    assert ac._resolve_codex_credential_and_base()[0] is None           # model-less stays conservative
    assert ac._select_pool_entry("openai-codex", AUX_MODEL)[1] is not None


def test_codex_aux_client_builds_for_unbenched_model(root, monkeypatch):
    from agent import auxiliary_client as ac
    from agent.credential_pool import load_pool

    monkeypatch.setattr(ac, "_read_codex_singleton_token", lambda: None)
    _bench(load_pool("openai-codex"))
    client, model = ac._build_codex_client(AUX_MODEL)
    assert client is not None and model == AUX_MODEL
    assert ac._build_codex_client(BENCHED) == (None, None)


def test_flush_from_a_stale_process_does_not_resurrect_a_reset_model_bench(root):
    """`hermes auth reset` in process B must survive process A's next ordinary flush (cf. #89415)."""
    from agent.credential_pool import load_pool

    live = load_pool("openai-codex")                       # process A: long-lived, holds the bench in memory
    _bench(live)
    assert load_pool("openai-codex").reset_statuses() == 1  # process B: `hermes auth reset openai-codex`
    assert _disk_cooldowns(root) == {}

    live._persist()                                        # A's next ordinary flush
    assert _disk_cooldowns(root) == {}
    assert load_pool("openai-codex").select() is not None  # a fresh model-less reader is no longer blocked


def test_a_bench_newer_than_the_reset_still_persists(root):
    """The reset marker must only outrank OLDER cooldowns, or one reset would make the entry immune."""
    from agent.credential_pool import load_pool

    load_pool("openai-codex").reset_statuses()             # stamp status_cleared_at
    live = load_pool("openai-codex")                       # loaded AFTER the reset
    _bench(live)
    live._persist()
    assert set(_disk_cooldowns(root)) == {BENCHED}
    assert load_pool("openai-codex").select(model=BENCHED) is None
