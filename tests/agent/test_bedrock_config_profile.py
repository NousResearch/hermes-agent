"""config.yaml ``bedrock.profile`` names the AWS profile Bedrock clients sign with.

The setting used to be ignored: only ``AWS_PROFILE`` took effect, so SSO / cross-account profiles
needed an env var. These tests build real botocore clients from a temp AWS config file (static
dummy keys, IMDS off) and read the credentials and region the client actually carries; no request
leaves the process.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import bedrock_adapter
from agent.bedrock_adapter import _get_bedrock_runtime_client, discover_bedrock_models, resolve_bedrock_region

REGION = "us-east-1"


@pytest.fixture
def aws_profiles(tmp_path, monkeypatch):
    """Two named profiles in a temp AWS config, ambient env keys that must NOT win when one is named."""
    aws_config = tmp_path / "aws-config"
    aws_config.write_text(
        "[profile sso-a]\naws_access_key_id = AKIA-PROFILE-A\naws_secret_access_key = a-secret\n"
        "region = ap-northeast-1\n"
        "[profile sso-b]\naws_access_key_id = AKIA-PROFILE-B\naws_secret_access_key = b-secret\n",
        encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(aws_config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-AMBIENT")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    bedrock_adapter.reset_client_cache()
    yield
    bedrock_adapter.reset_client_cache()


def _use_profile(monkeypatch, profile):
    cfg = {"bedrock": {"profile": profile}} if profile else {}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)


def _signing_key(client):
    return client._request_signer._credentials.access_key


def test_runtime_client_signs_with_the_configured_profile(aws_profiles, monkeypatch):
    pytest.importorskip("boto3")
    _use_profile(monkeypatch, "sso-a")
    assert _signing_key(_get_bedrock_runtime_client(REGION)) == "AKIA-PROFILE-A"
    # Switching the profile in the same region must not reuse the client built under the first one.
    _use_profile(monkeypatch, "sso-b")
    assert _signing_key(_get_bedrock_runtime_client(REGION)) == "AKIA-PROFILE-B"
    _use_profile(monkeypatch, "")
    assert _signing_key(_get_bedrock_runtime_client(REGION)) == "AKIA-AMBIENT"


def test_region_comes_from_the_configured_profile_unless_env_names_one(aws_profiles, monkeypatch):
    pytest.importorskip("botocore.session")
    _use_profile(monkeypatch, "sso-a")
    assert resolve_bedrock_region({}) == "ap-northeast-1"
    assert resolve_bedrock_region({"AWS_REGION": "eu-west-1"}) == "eu-west-1"


def test_discovery_cache_is_per_profile(monkeypatch):
    bedrock_adapter.reset_discovery_cache()
    control = MagicMock()
    control.list_foundation_models.return_value = {"modelSummaries": []}
    control.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}
    with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=control):
        for profile in ("sso-a", "sso-b", "sso-a"):
            _use_profile(monkeypatch, profile)
            discover_bedrock_models(REGION)
    assert control.list_foundation_models.call_count == 2


def test_bedrock_profile_is_a_recognized_config_key():
    """``hermes config set bedrock.profile <name>`` must not be refused as a typo of a key the adapter reads."""
    from hermes_cli.config import _validate_config_key
    assert _validate_config_key("bedrock.profile") == (True, None)
