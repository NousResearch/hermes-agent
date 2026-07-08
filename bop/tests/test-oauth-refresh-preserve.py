#!/usr/bin/env python3
import asyncio
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.mcp_oauth import HermesTokenStorage  # noqa: E402

try:
    from mcp.shared.auth import OAuthToken  # noqa: E402
except ImportError:
    class OAuthToken:
        def __init__(
            self,
            access_token,
            token_type="Bearer",
            expires_in=None,
            refresh_token=None,
            scope=None,
        ):
            self.access_token = access_token
            self.token_type = token_type
            self.expires_in = expires_in
            self.refresh_token = refresh_token
            self.scope = scope

        def model_dump(self, mode="json", exclude_none=True):
            data = {
                "access_token": self.access_token,
                "token_type": self.token_type,
                "expires_in": self.expires_in,
                "refresh_token": self.refresh_token,
                "scope": self.scope,
            }
            if exclude_none:
                data = {key: value for key, value in data.items() if value is not None}
            return data


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_mode_0600(path):
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected {path} mode 0600, got {mode:o}"


async def test_preserved_when_omitted():
    storage = HermesTokenStorage("gmail")
    path = storage._tokens_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": "old-access",
                "token_type": "Bearer",
                "expires_in": 100,
                "refresh_token": "old-refresh",
            }
        ),
        encoding="utf-8",
    )

    await storage.set_tokens(
        OAuthToken(
            access_token="new-access",
            token_type="Bearer",
            expires_in=3600,
        )
    )

    stored = read_json(path)
    assert stored["access_token"] == "new-access"
    assert stored["refresh_token"] == "old-refresh"
    assert "expires_at" in stored
    assert_mode_0600(path)


async def test_replaced_when_present():
    storage = HermesTokenStorage("gmail")
    path = storage._tokens_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": "old-access",
                "token_type": "Bearer",
                "expires_in": 100,
                "refresh_token": "old-refresh",
            }
        ),
        encoding="utf-8",
    )

    await storage.set_tokens(
        OAuthToken(
            access_token="new-access",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="new-refresh",
        )
    )

    stored = read_json(path)
    assert stored["access_token"] == "new-access"
    assert stored["refresh_token"] == "new-refresh"
    assert "expires_at" in stored
    assert_mode_0600(path)


async def main():
    original_home = os.environ.get("HERMES_HOME")
    with tempfile.TemporaryDirectory(prefix="hermes-oauth-refresh-preserve-") as tmp:
        os.environ["HERMES_HOME"] = tmp
        await test_preserved_when_omitted()
        await test_replaced_when_present()

    if original_home is None:
        os.environ.pop("HERMES_HOME", None)
    else:
        os.environ["HERMES_HOME"] = original_home

    print("oauth refresh-token preservation tests passed")


if __name__ == "__main__":
    asyncio.run(main())
