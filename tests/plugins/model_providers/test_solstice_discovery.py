"""Provider discovery must not require Solstice's inference dependencies."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_solstice_discovery_and_cua_probe_without_httpx(tmp_path):
    """PM carries YAML support but intentionally does not carry httpx."""
    repo = Path(__file__).resolve().parents[3]
    code = textwrap.dedent("""
        import importlib.abc
        import sys

        sys.path.insert(0, sys.argv[1])

        class MissingHttpx(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "httpx" or fullname.startswith("httpx."):
                    raise ModuleNotFoundError("No module named 'httpx'", name=fullname)

        assert "httpx" not in sys.modules
        sys.meta_path.insert(0, MissingHttpx())

        from pm.packages import CuaDriver
        env = CuaDriver()._probe_env()
        assert env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] == "0"

        from providers import get_provider_profile
        profile = get_provider_profile("solstice")
        assert profile is not None, "Solstice metadata must load without httpx"
        assert get_provider_profile("solstice-oauth") is profile
        assert "httpx" not in sys.modules
    """)
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               HERMES_HOME=str(tmp_path / "hermes"))
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(repo)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Failed to load bundled provider plugin solstice" not in result.stderr


@pytest.mark.parametrize("base_url", [None, "https://gateway.example.invalid/v1alpha"])
def test_solstice_client_creation_preserves_endpoint_and_bearer_auth(base_url):
    from providers import get_provider_profile

    profile = get_provider_profile("solstice")
    assert profile is not None
    client = profile.create_client(
        api_key="test-access-token", base_url=base_url,
        default_headers={"X-Test": "kept"}, unsupported_kwarg="ignored",
    )
    assert client is not None
    with client:
        from agent.gemini_native_adapter import GeminiNativeClient

        assert isinstance(client, GeminiNativeClient)
        assert client.base_url == (base_url or profile.base_url)
        assert client._auth_headers() == {"Authorization": "Bearer test-access-token"}
        assert client._headers()["X-Test"] == "kept"
        assert client.GENERATE_METHOD == "generateContentPerUserQuota"
        assert client.STREAM_METHOD == "streamGenerateContentPerUserQuota"
