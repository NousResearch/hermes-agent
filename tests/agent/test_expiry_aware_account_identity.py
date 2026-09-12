"""Use the existing OAuth JWT provenance without requiring hand-edited metadata."""
import base64
import json
from agent.credential_pool import PooledCredential
from agent.account_usage import _entry_codex_account_id


def test_standard_codex_pool_entry_exposes_account_identity():
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "account-a"}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    entry = PooledCredential("openai-codex", "a", "a", "oauth", 0, "manual", "e30." + payload + ".sig")
    assert entry.account_id == "account-a"
    assert _entry_codex_account_id(entry) == "account-a"
    entry.extra["account_id"] = "another-account"
    assert entry.account_id is None
    assert _entry_codex_account_id(entry) is None
