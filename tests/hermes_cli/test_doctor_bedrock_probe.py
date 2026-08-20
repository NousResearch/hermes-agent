"""Tests for the module-level Bedrock connectivity probe in ``hermes doctor``.

The probe checks the Bedrock *control plane* (``ListFoundationModels``) and
its green tick used to read as "inference works", while every request to the
configured inference endpoint could still fail (#87195). The success line
must say ``control plane only``.
"""

import sys
import types

import hermes_cli.doctor as doctor


def _patch_bedrock_auth(monkeypatch, has_creds=True):
    import agent.bedrock_adapter as ba

    monkeypatch.setattr(ba, "has_aws_credentials", lambda: has_creds)
    monkeypatch.setattr(ba, "resolve_aws_auth_env_var", lambda: "AWS_BEARER_TOKEN_BEDROCK")
    monkeypatch.setattr(ba, "resolve_bedrock_region", lambda: "us-east-1")


def _install_fake_boto3(monkeypatch, models=122, client_error=None):
    if client_error is None:
        fake_client = types.SimpleNamespace(
            list_foundation_models=lambda: {"modelSummaries": [{}] * models},
        )
    else:
        def _explode():
            raise client_error

        fake_client = types.SimpleNamespace(list_foundation_models=_explode)
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service_name, region_name=None, config=None: fake_client
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)


class TestBedrockProbe:
    def test_success_line_names_control_plane(self, monkeypatch):
        _patch_bedrock_auth(monkeypatch)
        _install_fake_boto3(monkeypatch, models=122)

        result = doctor._probe_bedrock()

        assert result.label == "AWS Bedrock"
        assert len(result.lines) == 1
        icon, label, detail = result.lines[0]
        assert label.strip() == "AWS Bedrock"
        assert "AWS_BEARER_TOKEN_BEDROCK" in detail
        assert "us-east-1" in detail
        assert "122 models" in detail
        assert "control plane only" in detail
        assert result.issues == []

    def test_no_aws_credentials_yields_no_lines(self, monkeypatch):
        _patch_bedrock_auth(monkeypatch, has_creds=False)

        result = doctor._probe_bedrock()

        assert result.lines == []
        assert result.issues == []

    def test_missing_boto3_reports_install_hint(self, monkeypatch):
        _patch_bedrock_auth(monkeypatch)
        monkeypatch.setitem(sys.modules, "boto3", None)

        result = doctor._probe_bedrock()

        assert len(result.lines) == 1
        icon, label, detail = result.lines[0]
        assert "boto3 not installed" in detail
        assert any("pip install boto3" in issue for issue in result.issues)

    def test_probe_failure_names_the_permission(self, monkeypatch):
        _patch_bedrock_auth(monkeypatch)
        _install_fake_boto3(monkeypatch, client_error=RuntimeError("boom"))

        result = doctor._probe_bedrock()

        assert len(result.lines) == 1
        icon, label, detail = result.lines[0]
        assert "RuntimeError" in detail
        assert any("ListFoundationModels" in issue for issue in result.issues)

    def test_broken_adapter_import_is_not_a_silent_pass(self, monkeypatch):
        """A user WITH credentials but a broken agent.bedrock_adapter install
        must see a warning, not an empty result indistinguishable from
        'no AWS credentials configured' (a silent green pass)."""
        monkeypatch.setitem(sys.modules, "agent.bedrock_adapter", None)

        result = doctor._probe_bedrock()

        assert len(result.lines) == 1
        icon, label, detail = result.lines[0]
        assert label.strip() == "AWS Bedrock"
        assert "bedrock adapter unavailable" in detail
        assert any(
            "agent.bedrock_adapter failed to import" in issue
            for issue in result.issues
        )
