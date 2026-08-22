"""#88989: resolve the env-var name to persist as model.key_env."""

from hermes_cli.model_switch import resolve_switch_key_env


def test_builtin_deepseek_uses_registry_env_var():
    assert resolve_switch_key_env("deepseek") == "DEEPSEEK_API_KEY"


def test_named_providers_dict_key_env_wins():
    assert (
        resolve_switch_key_env(
            "sensenova",
            user_providers={
                "sensenova": {
                    "base_url": "https://api.sensenova.cn/v1",
                    "key_env": "AGNES_API_KEY",
                }
            },
        )
        == "AGNES_API_KEY"
    )


def test_named_providers_dict_expands_env_ref():
    assert (
        resolve_switch_key_env(
            "custom:neuralwatt",
            user_providers={
                "neuralwatt": {"api_key": "${NEURALWATT_API_KEY}"},
            },
        )
        == "NEURALWATT_API_KEY"
    )


def test_custom_providers_list_key_env():
    assert (
        resolve_switch_key_env(
            "custom:agnes",
            custom_providers=[
                {
                    "name": "agnes",
                    "provider_key": "agnes",
                    "base_url": "https://example.test/v1",
                    "key_env": "AGNES_API_KEY",
                }
            ],
        )
        == "AGNES_API_KEY"
    )


def test_unknown_provider_clears():
    assert resolve_switch_key_env("not-a-real-provider") == ""
    assert resolve_switch_key_env("") == ""
