"""OAuth admission must treat non-list env declarations as undeclared (#151208)."""

import pytest

from hermes_cli.plugin_oauth import oauth_admission_problems


@pytest.mark.parametrize("field, malformed", [
    ("requires_env", 1),
    ("requires_env", True),
    ("requires_env", {"OWN_CLIENT_ID": {}}),
    ("optional_env", {"OWN_CLIENT_ID": {}}),
], ids=["number", "boolean", "mapping", "optional-mapping"])
def test_oauth_admission_requires_list_env_declarations(tmp_path, field, malformed):
    manifest = {
        "name": "native",
        "kind": "platform",
        field: malformed,
        "oauth": [{
            "id": "rooms",
            "authorize_url": "https://login.example.com/authorize",
            "token_url": "https://login.example.com/token",
            "scopes": ["read"],
            "hosts": ["api.example.com"],
            "client_id_env": "OWN_CLIENT_ID",
        }],
    }
    problems = oauth_admission_problems(manifest, tmp_path / "plugins" / "native")
    assert any("OWN_CLIENT_ID must be declared under requires_env or optional_env" in p for p in problems)
