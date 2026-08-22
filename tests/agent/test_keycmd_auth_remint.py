"""``key_cmd`` providers re-mint once on an auth failure.

A ``key_cmd`` provider has no credential-pool entry, so
``recover_with_credential_pool`` used to return at ``pool is None`` and no
recovery path existed: the cached bearer was returned until its advertised TTL
expired, even after the gateway rejected it.
"""

from __future__ import annotations

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.command_token_source import build_command_token_provider
from agent.error_classifier import FailoverReason


def _counting_source(tmp_path, label="demo"):
    counter = tmp_path / "n"
    counter.write_text("0")
    cmd = (
        f"python3 -c \"import pathlib;p=pathlib.Path(r'{counter}');"
        "n=int(p.read_text());p.write_text(str(n+1));"
        "print('tok-%d' % n)\""
    )
    return build_command_token_provider(cmd, label), counter


def _agent(api_key, provider="custom"):
    class _A:
        pass

    a = _A()
    a._credential_pool = None
    a.provider = provider
    a.base_url = "https://gateway.invalid/v1"
    a.api_key = api_key
    a.log_prefix = ""
    return a


class TestKeyCmdAuthRemint:
    def test_auth_failure_invalidates_the_cached_token(self, tmp_path):
        src, counter = _counting_source(tmp_path)
        agent = _agent(src)
        assert agent.api_key() == "tok-0"

        recovered, _ = recover_with_credential_pool(
            agent, status_code=401, has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )

        assert recovered is True
        assert agent.api_key() == "tok-1", "retry sends a freshly minted token"
        assert counter.read_text() == "2"

    def test_auth_permanent_also_recovers(self, tmp_path):
        """The classifier reports a failed key_cmd as auth_permanent."""
        src, _ = _counting_source(tmp_path)
        agent = _agent(src)
        agent.api_key()
        recovered, _ = recover_with_credential_pool(
            agent, status_code=None, has_retried_429=False,
            classified_reason=FailoverReason.auth_permanent,
        )
        assert recovered is True

    def test_bare_401_without_a_classification_recovers(self, tmp_path):
        src, _ = _counting_source(tmp_path)
        agent = _agent(src)
        agent.api_key()
        recovered, _ = recover_with_credential_pool(
            agent, status_code=401, has_retried_429=False,
        )
        assert recovered is True

    def test_only_one_remint_per_provider(self, tmp_path):
        """A revoked credential must not spin: the second attempt refuses."""
        src, counter = _counting_source(tmp_path)
        agent = _agent(src)
        agent.api_key()

        first, _ = recover_with_credential_pool(
            agent, status_code=401, has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )
        agent.api_key()
        second, _ = recover_with_credential_pool(
            agent, status_code=401, has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )

        assert (first, second) == (True, False)
        assert counter.read_text() == "2", "helper ran twice, not three times"

    @pytest.mark.parametrize(
        "reason",
        [
            FailoverReason.rate_limit,
            FailoverReason.timeout,
            FailoverReason.server_error,
            FailoverReason.context_overflow,
        ],
    )
    def test_non_auth_reasons_do_not_remint(self, tmp_path, reason):
        src, counter = _counting_source(tmp_path)
        agent = _agent(src)
        agent.api_key()
        recovered, _ = recover_with_credential_pool(
            agent, status_code=429, has_retried_429=False,
            classified_reason=reason,
        )
        assert recovered is False
        assert counter.read_text() == "1", "helper not re-run"

    @pytest.mark.parametrize("api_key", ["static-key", None, 123])
    def test_providers_without_a_key_cmd_are_untouched(self, api_key):
        """No invalidate() attribute means nothing to re-mint."""
        agent = _agent(api_key)
        recovered, _ = recover_with_credential_pool(
            agent, status_code=401, has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )
        assert recovered is False

    def test_a_raising_invalidate_does_not_break_recovery(self, tmp_path):
        class Exploding:
            def __call__(self):
                return "tok"

            def invalidate(self):
                raise RuntimeError("boom")

        agent = _agent(Exploding())
        recovered, _ = recover_with_credential_pool(
            agent, status_code=401, has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )
        assert recovered is False
