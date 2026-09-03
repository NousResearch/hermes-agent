"""Cooldown-aware runtime resolution for pooled API-key providers.

A 429 benches every credential for a provider in ``auth.json`` with an
expiring TTL that every process shares.  ``pool.select()`` honours that bench,
but the secret resolver's credential-pool fallback iterates entries without
consulting status, so resolution used to hand back a benched key: the request
went out, 429'd, and only then did the fallback chain run -- one wasted
round-trip per turn, forever, because a fresh agent is built per gateway
message and always starts from the configured primary.

These exercise the real chain (real ``auth.json``, real ``load_pool``, real
``resolve_runtime_provider``) against a temp ``HERMES_HOME``.  Nothing is
mocked between the credential store and the resolver.
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY

# Gemini keys carry a declared "AIza" prefix; the secret resolver skips a
# malformed value, so the fixture has to look like the real thing.
_KEY = "AIzaSyTestKeyForCooldownResolution"


def _entry(
    cred_id: str,
    *,
    exhausted_age: float | None,
    priority: int = 0,
    error_code: int = 429,
) -> dict:
    entry = {
        "id": cred_id,
        "label": cred_id,
        "auth_type": "api_key",
        "priority": priority,
        "source": "manual",
        "access_token": f"{_KEY}{priority:06d}",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    }
    if exhausted_age is not None:
        entry["last_status"] = "exhausted"
        entry["last_status_at"] = time.time() - exhausted_age
        entry["last_error_code"] = error_code
    return entry


def _home(tmp_path, monkeypatch, entries: list[dict], *, with_fallback: bool = True):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"gemini": entries}}, indent=2),
        encoding="utf-8",
    )

    config = ["model:", "  default: gemini-3.7-flash", "  provider: gemini"]
    if with_fallback:
        config += [
            "fallback_providers:",
            "  - provider: openai-codex",
            "    model: gpt-5.6-luna",
        ]
    (hermes_home / "config.yaml").write_text("\n".join(config) + "\n", encoding="utf-8")
    return hermes_home


def test_all_pooled_credentials_cooling_down_defers_to_the_fallback_chain(
    tmp_path, monkeypatch
):
    """Every key benched by a recent 429 -> say so on the resolved runtime.

    Resolution still succeeds; the annotation is what lets a caller route
    around the cooldown instead of spending a request that will 429.
    """
    _home(
        tmp_path,
        monkeypatch,
        [_entry(f"cred-{i}", exhausted_age=300, priority=i) for i in range(3)],
    )

    resolved = rp.resolve_runtime_provider(requested="gemini")

    cooling_until = resolved.get(CREDENTIALS_COOLING_DOWN_KEY)
    assert cooling_until is not None
    assert cooling_until > time.time()
    # The runtime stays usable: this is a demotion the caller routes around,
    # not an auth failure that blocks resolution.
    assert resolved["api_key"].startswith(_KEY)


def test_a_real_429_records_the_cooldown_that_the_next_process_skips(
    tmp_path, monkeypatch
):
    """The three stages end to end: record -> skip -> return after expiry.

    Stage 1 goes through the same call the turn loop makes on a 429
    (``mark_exhausted_and_rotate``) rather than hand-written status, so the
    recording half is covered too. Stages 2 and 3 then read that record back
    through a freshly loaded pool -- the cross-process path.
    """
    from agent.credential_pool import load_pool

    _home(tmp_path, monkeypatch, [_entry("cred-0", exhausted_age=None)])

    # Stage 1 — a 429 arrives and the pool records the bench.
    pool = load_pool("gemini")
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"message": "Gemini HTTP 429 (RESOURCE_EXHAUSTED)"},
        credential_id="cred-0",
    )
    stored = json.loads((tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8"))
    assert stored["credential_pool"]["gemini"][0]["last_status"] == "exhausted"
    assert stored["credential_pool"]["gemini"][0]["last_error_code"] == 429

    # Stage 2 — a later process reads that record and flags the key as benched.
    flagged = rp.resolve_runtime_provider(requested="gemini")
    assert flagged.get(CREDENTIALS_COOLING_DOWN_KEY) is not None

    # Stage 3 — the bench expires on its own and the provider comes back.
    path = tmp_path / "hermes" / "auth.json"
    aged = json.loads(path.read_text(encoding="utf-8"))
    for row in aged["credential_pool"]["gemini"]:
        row["last_status_at"] = time.time() - 3 * 3600
    path.write_text(json.dumps(aged, indent=2), encoding="utf-8")

    resolved = rp.resolve_runtime_provider(requested="gemini")
    assert resolved.get(CREDENTIALS_COOLING_DOWN_KEY) is None
    assert resolved["provider"] == "gemini"
    assert resolved["api_key"].startswith(_KEY)


def test_non_429_bench_is_not_reported_as_a_rate_limit(tmp_path, monkeypatch):
    """A pool benched by revoked keys must not be dressed up as a quota cap.

    ``next_available_at()`` keys off the exhausted status alone, which also
    covers 401/402/403/5xx. Reporting those as ``credentials_cooling_down``
    would mark them rate-limited, so the operator would never be prompted to
    re-authenticate and the gateway would log "rate-limited (429)" for an auth
    failure (#32790).
    """
    _home(
        tmp_path,
        monkeypatch,
        [_entry(f"cred-{i}", exhausted_age=60, priority=i, error_code=401) for i in range(2)],
    )

    resolved = rp.resolve_runtime_provider(requested="gemini")

    assert resolved.get(CREDENTIALS_COOLING_DOWN_KEY) is None
    assert resolved["api_key"].startswith(_KEY)


def test_mixed_bench_reasons_do_not_trigger_the_gate(tmp_path, monkeypatch):
    """One non-429 bench in the pool is enough to disqualify the rate-limit claim."""
    _home(
        tmp_path,
        monkeypatch,
        [
            _entry("cred-0", exhausted_age=300, priority=0, error_code=429),
            _entry("cred-1", exhausted_age=300, priority=1, error_code=402),
        ],
    )

    resolved = rp.resolve_runtime_provider(requested="gemini")

    assert resolved.get(CREDENTIALS_COOLING_DOWN_KEY) is None
    assert resolved["api_key"].startswith(_KEY)


def test_env_backed_key_is_still_recognised_as_pool_backed(tmp_path, monkeypatch):
    """An env var wins over the pool inside the secret resolver.

    So a key that is ALSO a pool entry resolves with an env-var ``source``
    while its pool entry cools down. Matching on the ``source`` stamp alone
    would miss that and send the doomed request anyway; the gate matches the
    resolved secret back to an entry instead.
    """
    entries = [
        _entry("cred-0", exhausted_age=300, priority=0),
        _entry("cred-1", exhausted_age=300, priority=1),
    ]
    entries[0]["source"] = "env:GOOGLE_API_KEY"
    _home(tmp_path, monkeypatch, entries)
    monkeypatch.setenv("GOOGLE_API_KEY", entries[0]["access_token"])

    from hermes_cli.auth import resolve_api_key_provider_credentials

    # Precondition: resolution really does come from the env var here, so the
    # source-prefix check alone would not fire.
    assert resolve_api_key_provider_credentials("gemini")["source"] == "GOOGLE_API_KEY"

    resolved = rp.resolve_runtime_provider(requested="gemini")
    assert resolved.get(CREDENTIALS_COOLING_DOWN_KEY) is not None


def test_cooldown_flag_never_blocks_resolution(tmp_path, monkeypatch):
    """Status probes, model pickers, one-shot setup and readiness checks share
    this resolver. Reporting the cooldown must not turn a configured provider
    into a resolution failure for them — the flag is additive, and the runtime
    they get back is the same one they got before.
    """
    _home(
        tmp_path,
        monkeypatch,
        [_entry(f"cred-{i}", exhausted_age=300, priority=i) for i in range(3)],
    )

    resolved = rp.resolve_runtime_provider(requested="gemini")

    assert resolved["provider"] == "gemini"
    assert resolved["base_url"]
    assert resolved["api_key"].startswith(_KEY)


def test_available_credential_resolves_untouched(tmp_path, monkeypatch):
    """One healthy key in the pool -> the gate must stay out of the way."""
    _home(
        tmp_path,
        monkeypatch,
        [
            _entry("cred-0", exhausted_age=300, priority=0),
            _entry("cred-1", exhausted_age=None, priority=1),
        ],
    )

    resolved = rp.resolve_runtime_provider(requested="gemini")

    assert resolved.get(CREDENTIALS_COOLING_DOWN_KEY) is None
    assert resolved["provider"] == "gemini"
    assert resolved["api_key"].startswith(_KEY)
