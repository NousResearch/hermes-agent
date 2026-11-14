"""``hermes doctor`` on an EC2 host whose Bedrock credential is the instance role (#153434).

Doctor disables boto3's instance-metadata lookup so non-AWS users don't wait on 169.254.169.254. When
Bedrock is the configured provider, that lookup is how an instance-role host finds its credential: with it
disabled, a working host reported ``NoCredentialsError`` and was told to check its IAM permissions.
"""

from __future__ import annotations

import os
import sys
import types

from hermes_cli import doctor
from hermes_cli import doctor_connectivity as dc
from hermes_constants import get_hermes_home


class NoCredentialsError(Exception):
    """Stand-in for botocore's error: the probe reports the exception's class name."""


def _install_fake_boto3(monkeypatch):
    """boto3 whose credential chain only succeeds when the EC2 metadata lookup is allowed (instance role)."""
    def client(service, region_name=None, config=None):
        if os.environ.get("AWS_EC2_METADATA_DISABLED", "").lower() == "true":
            raise NoCredentialsError("Unable to locate credentials")
        return types.SimpleNamespace(list_foundation_models=lambda: {"modelSummaries": [{}, {}, {}]})

    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = lambda **kwargs: kwargs
    botocore = types.ModuleType("botocore")
    botocore.config = botocore_config
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)


def _instance_role_host(monkeypatch, provider: str):
    # The issue's setup: AWS_PROFILE names a region-only profile; the credential itself comes from IMDS.
    monkeypatch.setenv("AWS_PROFILE", "x")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    for var in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_EC2_METADATA_DISABLED"):
        monkeypatch.delenv(var, raising=False)  # conftest disables IMDS for the whole suite; a real shell doesn't
    (get_hermes_home() / "config.yaml").write_text(  # the suite's temp HERMES_HOME
        f"model:\n  provider: {provider}\n  default: anthropic.claude-sonnet-5\n", encoding="utf-8")


def test_doctor_reports_bedrock_healthy_on_an_instance_role_host(monkeypatch, capsys):
    _install_fake_boto3(monkeypatch)
    _instance_role_host(monkeypatch, provider="aws")  # an alias of bedrock
    monkeypatch.setattr(doctor, "build_probes", lambda: [("AWS Bedrock", dc._probe_bedrock)])

    finding = doctor._check_api_connectivity(False)

    out = capsys.readouterr().out
    assert "NoCredentialsError" not in out
    assert "✓" in out and "(AWS_PROFILE, us-east-1, 3 models)" in out
    assert finding.issues == []
    assert "AWS_EC2_METADATA_DISABLED" not in os.environ


def test_doctor_keeps_the_metadata_lookup_off_when_bedrock_is_not_the_provider(monkeypatch):
    _instance_role_host(monkeypatch, provider="openrouter")
    seen = []

    def probe():
        seen.append(os.environ.get("AWS_EC2_METADATA_DISABLED"))
        return dc._row("stand-in", "ok")

    monkeypatch.setattr(doctor, "build_probes", lambda: [("stand-in", probe)])

    doctor._check_api_connectivity(False)

    assert seen == ["true"]  # no multi-second IMDS timeout for users who don't run on Bedrock
