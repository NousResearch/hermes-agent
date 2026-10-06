"""A ``blocked_config`` credential verdict names the profile + HERMES_HOME the scheduler actually
read (#116213): "No Codex credentials stored" from a gateway whose home differs from the shell that
works is otherwise indistinguishable from a genuine login gap."""

import json
import re

import pytest

from cron.scheduler_preflight import _preflight_check_provider_key
from cron.scheduler_provider import _profile_cron_scope

JOB = {"id": "7a6ae427c1d8", "name": "radar", "provider": "openai-codex", "model": "gpt-5.6-sol"}


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    root = tmp_path / "root"
    alpha = root / "profiles" / "alpha"
    alpha.mkdir(parents=True)
    for home in (root, alpha):
        (home / "config.yaml").write_text("model:\n  default: gpt-5.6-sol\n  provider: openai-codex\n")
    (root / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    return root, alpha


def _scope(reason: str) -> tuple:
    match = re.search(r"\[profile '([^']+)', HERMES_HOME (.+?)\]", reason)
    assert match, reason
    return match.group(1), match.group(2)


def test_missing_codex_credential_verdict_names_the_home_it_read(two_homes):
    root, alpha = two_homes

    reason = _preflight_check_provider_key(JOB, {"cron": {}})
    assert reason
    assert _scope(reason) == ("default", str(root))

    # Multiplex tick of a satellite profile: the verdict names alpha, not the gateway's launch home.
    with _profile_cron_scope(alpha):
        reason = _preflight_check_provider_key(JOB, {"cron": {}})
    assert reason and _scope(reason) == ("alpha", str(alpha))


def test_halted_fallback_chain_does_not_skip_primary_credential_preflight(monkeypatch):
    calls = []
    chain = [{"provider": "openrouter", "model": "fallback-model"}]
    monkeypatch.setattr(
        "hermes_cli.fallback_config.fallback_halt_active", lambda: (True, "halted")
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: calls.append(kwargs),
    )

    cfg = {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
        "fallback_providers": chain,
    }
    assert _preflight_check_provider_key({"id": "unpinned"}, cfg) is None
    assert len(calls) == 1
    assert calls[0]["requested"] != "openrouter"
    assert calls[0]["target_model"] != "fallback-model"
