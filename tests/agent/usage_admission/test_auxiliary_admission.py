"""Auxiliary client paths honour Codex usage admission."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestAuxiliaryUsageAdmission:
    def test_auth_refresh_policy_denial_selects_sibling_without_exhausting(self):
        auth_err = Exception("unauthorized")
        auth_err.status_code = 401
        replacement = SimpleNamespace(id="cred-b")

        class _Pool:
            def has_credentials(self):
                return True

            def try_refresh_matching_with_admission_status(
                self, *, api_key_hint=None, credential_id=None
            ):
                assert api_key_hint == "failed-token-a"
                assert credential_id is None
                return None, "cred-a"

            def try_refresh_current_with_admission_status(self):
                raise AssertionError("recovery must target the failed credential")

            def select(self):
                return replacement

            def mark_exhausted_and_rotate(self, **kwargs):
                raise AssertionError(
                    "policy denial must not mutate exhaustion state"
                )

        pool = _Pool()
        with (
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client._evict_cached_clients") as evict,
        ):
            from agent.auxiliary_client import _recover_provider_pool

            assert (
                _recover_provider_pool(
                    "openai-codex",
                    auth_err,
                    failed_api_key="failed-token-a",
                )
                is True
            )

        evict.assert_called_once_with("openai-codex")

    def test_auto_codex_keeps_unrelated_explicit_main_runtime_credential(self):
        explicit_token = "unrelated-explicit-token"
        pooled_entry = SimpleNamespace(
            id="configured-entry",
            provider="openai-codex",
            runtime_api_key="configured-pool-token",
            runtime_base_url="https://chatgpt.com/backend-api/codex",
        )
        pool = MagicMock()
        pool.has_credentials.return_value = True
        pool.select.return_value = pooled_entry
        client = MagicMock()

        import agent.auxiliary_client as aux

        with (
            patch(
                "agent.auxiliary_client.get_pool_usage_limits",
                return_value={"configured-entry": 80.0},
            ),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch(
                "agent.auxiliary_client.usage_admission_credential_denied",
                return_value=False,
            ) as denied,
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(client, "gpt-5.4"),
            ) as resolve,
        ):
            aux.shutdown_cached_clients()
            try:
                resolved, _model = aux._get_cached_client(
                    "auto",
                    "gpt-5.4",
                    main_runtime={
                        "provider": "openai-codex",
                        "model": "gpt-5.4",
                        "api_key": explicit_token,
                        "base_url": "https://chatgpt.com/backend-api/codex",
                        "api_mode": "codex_responses",
                    },
                )
            finally:
                aux.shutdown_cached_clients()

        assert resolved is client
        pool.select.assert_not_called()
        denied.assert_called_once_with("openai-codex", explicit_token)
        assert resolve.call_args.kwargs["explicit_api_key"] == explicit_token

    def test_auto_codex_rejects_matching_denied_explicit_main_runtime_credential(
        self,
    ):
        import agent.auxiliary_client as aux

        pool = MagicMock()
        pool.has_credentials.return_value = True
        with (
            patch(
                "agent.auxiliary_client.get_pool_usage_limits",
                return_value={"configured-entry": 80.0},
            ),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch(
                "agent.auxiliary_client.usage_admission_credential_denied",
                return_value=True,
            ) as denied,
            patch("agent.auxiliary_client.resolve_provider_client") as resolve,
        ):
            aux.shutdown_cached_clients()
            client, model = aux._get_cached_client(
                "auto",
                "gpt-5.4",
                main_runtime={
                    "provider": "openai-codex",
                    "model": "gpt-5.4",
                    "api_key": "matching-denied-token",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "api_mode": "codex_responses",
                },
            )

        assert client is None
        assert model is None
        pool.select.assert_not_called()
        denied.assert_called_once_with("openai-codex", "matching-denied-token")
        resolve.assert_not_called()
