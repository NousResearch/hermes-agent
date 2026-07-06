from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import yaml


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.read_timeout = kwargs.get("read_timeout")
        self.connect_timeout = kwargs.get("connect_timeout")
        self.retries = kwargs.get("retries")


def _install_fake_boto3_and_botocore(monkeypatch, calls):
    boto3_mod = ModuleType("boto3")
    boto3_mod.__version__ = "1.34.59"

    def _client(service_name, **kwargs):
        calls.append({"service_name": service_name, "kwargs": kwargs})
        return SimpleNamespace(service_name=service_name, kwargs=kwargs)

    boto3_mod.client = _client

    botocore_mod = ModuleType("botocore")
    config_mod = ModuleType("botocore.config")
    config_mod.Config = _FakeConfig
    botocore_mod.config = config_mod

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "botocore", botocore_mod)
    monkeypatch.setitem(sys.modules, "botocore.config", config_mod)


def test_bedrock_runtime_client_uses_botocore_config_from_config_yaml(monkeypatch):
    hermes_home = Path(os.environ["HERMES_HOME"])
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "bedrock": {
                    "read_timeout": 777,
                    "connect_timeout": 12,
                    "retries_max_attempts": 5,
                    "retries_mode": "standard",
                }
            }
        ),
        encoding="utf-8",
    )

    calls = []
    _install_fake_boto3_and_botocore(monkeypatch, calls)

    from agent import bedrock_adapter

    bedrock_adapter.reset_client_cache()
    try:
        client = bedrock_adapter._get_bedrock_runtime_client("us-west-2")
    finally:
        bedrock_adapter.reset_client_cache()

    assert client.service_name == "bedrock-runtime"
    assert calls[0]["service_name"] == "bedrock-runtime"
    assert calls[0]["kwargs"]["region_name"] == "us-west-2"

    config = calls[0]["kwargs"]["config"]
    assert isinstance(config, _FakeConfig)
    assert config.read_timeout == 777.0
    assert config.connect_timeout == 12.0
    assert config.retries == {"max_attempts": 5, "mode": "standard"}


def test_bedrock_control_client_uses_same_botocore_config(monkeypatch):
    calls = []
    _install_fake_boto3_and_botocore(monkeypatch, calls)

    from agent import bedrock_adapter

    bedrock_adapter.reset_client_cache()
    try:
        client = bedrock_adapter._get_bedrock_control_client("us-east-1")
    finally:
        bedrock_adapter.reset_client_cache()

    assert client.service_name == "bedrock"
    assert calls[0]["service_name"] == "bedrock"
    config = calls[0]["kwargs"]["config"]
    assert isinstance(config, _FakeConfig)
    assert config.read_timeout == 600.0
    assert config.connect_timeout == 10.0
    assert config.retries == {"max_attempts": 3, "mode": "adaptive"}
