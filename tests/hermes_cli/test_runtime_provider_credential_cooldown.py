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


# ---------------------------------------------------------------------------
# Provider shapes whose credentials come from a pool but whose resolution does
# NOT run through the API-key branch above: OpenRouter (whose fall-through
# reads env vars) and a named custom endpoint (whose pool lives under its own
# ``custom:<name>`` key). Both used to resolve with no cooldown reported at
# all, so a caller owning a fallback chain never learned to route around them.
# ---------------------------------------------------------------------------

# A sole credential is benched for EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS (60s),
# deliberately shorter than the rotating-pool hour, so these fixtures must sit
# well inside that window to still be cooling when the resolver looks.
_SOLE_BENCH_AGE = 5


def _benched_row(cred_id: str, key: str, source: str) -> dict:
    return {
        "id": cred_id,
        "label": cred_id,
        "auth_type": "api_key",
        "priority": 0,
        "source": source,
        "access_token": key,
        "last_status": "exhausted",
        "last_status_at": time.time() - _SOLE_BENCH_AGE,
        "last_error_code": 429,
    }


def _pool_home(tmp_path, monkeypatch, pool: dict, config: str):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for var in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "HERMES_INFERENCE_PROVIDER",
        "HERMES_INFERENCE_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": pool}, indent=2), encoding="utf-8"
    )
    (hermes_home / "config.yaml").write_text(config, encoding="utf-8")
    return hermes_home


def test_openrouter_reports_its_cooldown_instead_of_resolving_keyless(
    tmp_path, monkeypatch
):
    """OpenRouter's fall-through reads env vars, not the pool.

    With every pooled key benched, ``pool.select()`` refuses them all and
    resolution drops through to ``OPENROUTER_API_KEY`` -- unset here, as it is
    for anyone who registered their keys with ``hermes auth add``. The runtime
    used to come back keyless and un-annotated, so the caller reported "no API
    key configured" and never consulted a perfectly good fallback chain.
    """
    key = "sk-test-openrouter-" + "0" * 40
    _pool_home(
        tmp_path,
        monkeypatch,
        {"openrouter": [_benched_row("or-0", key, "manual")]},
        "model:\n  default: some/model\n  provider: openrouter\n",
    )

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    cooling_until = resolved.get(CREDENTIALS_COOLING_DOWN_KEY)
    assert cooling_until is not None
    assert cooling_until > time.time()


def test_a_named_custom_endpoint_reports_its_own_pools_cooldown(
    tmp_path, monkeypatch
):
    """A custom endpoint keeps its credentials under ``custom:<name>``.

    ``_try_resolve_from_custom_pool`` returns None both when an endpoint has
    no pool and when its pool refuses every benched entry -- so resolution
    falls through to the ``custom_providers`` api_key, which is the very
    credential the pool is cooling. Probing the bare ``custom`` pool would
    read an empty one and call it healthy.
    """
    key = "sk-test-custom-" + "0" * 32
    _pool_home(
        tmp_path,
        monkeypatch,
        {"custom:myllm": [_benched_row("cu-0", key, "config:myllm")]},
        "model:\n"
        "  default: my-model\n"
        "  provider: myllm\n"
        "custom_providers:\n"
        "  - name: myllm\n"
        "    base_url: https://llm.example.test/v1\n"
        f"    api_key: {key}\n"
        "    model: my-model\n",
    )

    resolved = rp.resolve_runtime_provider(requested="myllm")

    assert resolved["api_key"] == key
    cooling_until = resolved.get(CREDENTIALS_COOLING_DOWN_KEY)
    assert cooling_until is not None
    assert cooling_until > time.time()


def test_a_healthy_openrouter_pool_is_not_reported_as_cooling(
    tmp_path, monkeypatch
):
    """The negative half: one live key means the provider is usable.

    Guards the keyless branch added for the case above from swallowing every
    provider whose runtime happens to carry no inline key.
    """
    key = "sk-test-openrouter-live-" + "0" * 32
    row = _benched_row("or-live", key, "manual")
    row.pop("last_status")
    row.pop("last_error_code")
    _pool_home(
        tmp_path,
        monkeypatch,
        {"openrouter": [row]},
        "model:\n  default: some/model\n  provider: openrouter\n",
    )

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved.get(CREDENTIALS_COOLING_DOWN_KEY) is None
