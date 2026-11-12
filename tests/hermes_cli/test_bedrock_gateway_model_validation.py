"""Native Bedrock /model validation must not probe OpenAI-compatible /models."""
import unittest
from unittest.mock import patch

from hermes_cli.models_validate import validate_requested_model

MODELS = ["global.anthropic.claude-sonnet-5", "global.anthropic.claude-opus-5-5",
          "global.anthropic.claude-haiku-4-5-20251001-v1:0"]
URLS = ["https://bedrock-runtime.us-east-1.amazonaws.com",
        "https://custom-gateway.invalid/bedrock/tenant-a"]


class BedrockGatewayModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.entry = {"models": {m: {} for m in MODELS}, "discover_models": False}
        self.enterContext(patch("hermes_cli.config.load_config_readonly",
                               side_effect=lambda: {"providers": {"bedrock": self.entry}}))
        self.probe = self.enterContext(patch("hermes_cli.models.probe_api_models",
                                            side_effect=AssertionError("Unexpected /models probe")))
        self.live = self.enterContext(patch("hermes_cli.models.fetch_api_models",
                                           side_effect=AssertionError("Unexpected live model listing")))
        self.discovery = self.enterContext(patch("agent.bedrock_adapter.discover_bedrock_models",
                                                side_effect=AssertionError("Unexpected AWS discovery")))
        self.enterContext(patch("agent.bedrock_adapter.resolve_bedrock_runtime_region", return_value="us-east-1"))
        self.enterContext(patch("hermes_cli.models_validate._static_catalog", return_value=[]))

    def validate(self, model, provider="bedrock", url=URLS[1]):
        return validate_requested_model(model, provider, api_key="test-bearer", base_url=url,
                                        api_mode="bedrock_converse")

    def assert_offline(self):
        self.probe.assert_not_called()
        self.live.assert_not_called()
        self.discovery.assert_not_called()

    def test_three_configured_models_with_dict_and_list_on_both_runtime_urls(self):
        for catalog in [list(MODELS), {m: {} for m in MODELS}]:
            self.entry["models"] = catalog
            for url in URLS:
                for model in MODELS:
                    with self.subTest(catalog=type(catalog).__name__, url=url, model=model):
                        result = self.validate(model, url=url)
                        self.assertEqual(result, {"accepted": True, "persist": True,
                                                  "recognized": True, "message": None})
        self.assert_offline()

    def test_native_aliases_are_normalized_before_dispatch(self):
        for provider in ["aws", "aws-bedrock", "amazon-bedrock", "amazon", "BEDROCK"]:
            with self.subTest(provider=provider):
                self.assertTrue(self.validate(MODELS[0], provider=provider)["recognized"])
        self.assert_offline()

    def test_configured_models_win_before_enabled_discovery(self):
        self.entry["discover_models"] = True
        self.assertTrue(self.validate(MODELS[1])["recognized"])
        self.assert_offline()

    def test_unknown_model_without_discovery_is_unrecognized_soft_accept(self):
        result = self.validate("unlisted.example-model")
        self.assertTrue(result["accepted"])
        self.assertTrue(result["persist"])
        self.assertFalse(result["recognized"])
        self.assertIn("discovery is disabled", result["message"])
        self.assert_offline()

    def test_enabled_discovery_uses_bedrock_control_plane_only(self):
        self.entry["discover_models"] = True
        self.discovery.side_effect = None
        self.discovery.return_value = [{"id": "new.inference-profile"}]
        result = self.validate("new.inference-profile")
        self.assertTrue(result["recognized"])
        self.discovery.assert_called_once_with("us-east-1")
        self.probe.assert_not_called()
        self.live.assert_not_called()

    def test_failed_discovery_uses_offline_fallback_not_generic_live_probe(self):
        self.entry["discover_models"] = True
        self.discovery.side_effect = RuntimeError("control plane unavailable")
        result = self.validate("unlisted.example-model")
        self.assertTrue(result["accepted"])
        self.assertTrue(result["persist"])
        self.assertFalse(result["recognized"])
        self.discovery.assert_called_once()
        self.probe.assert_not_called()
        self.live.assert_not_called()

    def test_non_bedrock_custom_endpoint_still_probes(self):
        self.probe.side_effect = None
        self.probe.return_value = {"models": ["custom-model"]}
        result = self.validate("custom-model", provider="custom")
        self.assertTrue(result["recognized"])
        self.probe.assert_called_once()
        self.live.assert_not_called()
        self.discovery.assert_not_called()


if __name__ == "__main__":
    unittest.main()
