"""A pool-less ``key_cmd`` provider re-mints a rejected token once (#91119).

``recover_with_credential_pool`` used to return at ``pool is None``, so the cached bearer was
reused until its advertised TTL expired even after the gateway rejected it.
"""

from __future__ import annotations

import sys

import pytest

from agent import command_token_source
from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.command_token_source import CommandTokenError, build_command_token_provider
from agent.error_classifier import FailoverReason


def _counting_source(tmp_path, label="demo", name="n"):
    """A real helper that prints ``tok-<n>`` and counts its runs."""
    counter = tmp_path / name
    counter.write_text("0")
    script = (
        f"import pathlib; p = pathlib.Path(r'{counter}'); n = int(p.read_text()); "
        "p.write_text(str(n + 1)); print('tok-%d' % n)"
    )
    return build_command_token_provider(f'"{sys.executable}" -c "{script}"', label), counter


def _agent(api_key, provider="custom"):
    class _Agent:
        _credential_pool = None
        base_url = "https://gateway.invalid/v1"
        log_prefix = ""

    agent = _Agent()
    agent.provider, agent.api_key = provider, api_key
    return agent


def _recover(agent, status_code=401, reason=FailoverReason.auth):
    recovered, _ = recover_with_credential_pool(
        agent, status_code=status_code, has_retried_429=False, classified_reason=reason,
    )
    return recovered


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [(401, FailoverReason.auth), (None, FailoverReason.auth_permanent), (401, None)],
)
def test_a_rejected_token_is_re_minted_for_the_retry(tmp_path, status_code, reason):
    source, counter = _counting_source(tmp_path)
    agent = _agent(source)
    assert agent.api_key() == "tok-0"

    assert _recover(agent, status_code, reason) is True
    assert agent.api_key() == "tok-1", "the retry sends a freshly minted token"
    assert counter.read_text() == "2"


def test_a_re_minted_token_that_is_rejected_again_does_not_spin(tmp_path):
    source, counter = _counting_source(tmp_path)
    agent = _agent(source)
    agent.api_key()

    assert _recover(agent) is True
    agent.api_key()
    assert _recover(agent) is False
    assert agent.api_key() == "tok-1"
    assert counter.read_text() == "2", "the helper ran twice, not three times"


def test_the_next_scheduled_mint_re_arms_recovery(tmp_path, monkeypatch):
    """A credential that rotates again hours later can still recover without a restart."""
    source, counter = _counting_source(tmp_path)
    agent = _agent(source)
    agent.api_key()
    assert _recover(agent) is True
    agent.api_key()
    assert _recover(agent) is False

    real_monotonic = command_token_source.time.monotonic
    monkeypatch.setattr(command_token_source.time, "monotonic", lambda: real_monotonic() + 3600)
    assert agent.api_key() == "tok-2", "the no-TTL window expired, so the helper ran on schedule"
    assert _recover(agent) is True
    assert agent.api_key() == "tok-3"
    assert counter.read_text() == "4"


def test_the_bound_is_per_credential_not_per_provider_name(tmp_path):
    first, _ = _counting_source(tmp_path, name="first")
    second, second_counter = _counting_source(tmp_path, name="second")
    first_agent, second_agent = _agent(first, provider=""), _agent(second, provider="")
    first_agent.api_key(), second_agent.api_key()

    assert _recover(first_agent) is True
    assert _recover(second_agent) is True, "another source's re-mint must not use up this one"
    assert second_agent.api_key() == "tok-1"
    assert second_counter.read_text() == "2"


def test_a_failed_mint_is_not_retried(tmp_path):
    """The classifier reports a failed helper as auth_permanent; re-running it cannot help."""
    marker = tmp_path / "runs"
    source = build_command_token_provider(f"echo x >> {marker}; exit 1", "demo")
    agent = _agent(source)
    with pytest.raises(CommandTokenError):
        agent.api_key()

    assert _recover(agent, None, FailoverReason.auth_permanent) is False
    assert marker.read_text().count("x") == 1


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (403, None),
        (429, FailoverReason.rate_limit),
        (None, FailoverReason.timeout),
        (500, FailoverReason.server_error),
        (400, FailoverReason.context_overflow),
    ],
)
def test_other_failures_do_not_re_mint(tmp_path, status_code, reason):
    source, counter = _counting_source(tmp_path)
    agent = _agent(source)
    agent.api_key()

    assert _recover(agent, status_code, reason) is False
    assert agent.api_key() == "tok-0"
    assert counter.read_text() == "1"


class _DuckTypedSource:
    def __call__(self):
        return "tok"

    def invalidate(self):
        raise AssertionError("only a CommandTokenSource is invalidated")


@pytest.mark.parametrize("api_key", ["static-key", None, 123, _DuckTypedSource()])
def test_credentials_that_are_not_key_cmd_sources_are_untouched(api_key):
    assert _recover(_agent(api_key)) is False
