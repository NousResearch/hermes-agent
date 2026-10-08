"""``hermes model`` custom-endpoint setup must key each scope's ``.env`` slot on the scope.

The key was stored under a host[:port]-derived variable, so a second scope on the same
endpoint reused the first one's slot: the wizard wrote the new key over the old, both
entries pointed at one variable, and one subscription silently replaced the other
(#118285). The identity now comes from the display name — the scope's own name — with
host[:port] kept as the fallback for an endpoint saved without one.
"""

from __future__ import annotations

from hermes_cli.config import custom_endpoint_key_env
from hermes_cli.model_setup_flows_custom import _custom_endpoint_key_identity

URL = "https://opencode.ai/zen/v1"


def test_two_scopes_of_one_endpoint_get_two_env_slots():
    work = custom_endpoint_key_env(_custom_endpoint_key_identity(URL, "OpenCode Zen (work)"))
    personal = custom_endpoint_key_env(_custom_endpoint_key_identity(URL, "OpenCode Zen (personal)"))

    assert work == "HERMES_CUSTOM_OPENCODE_ZEN_WORK_API_KEY"
    assert personal == "HERMES_CUSTOM_OPENCODE_ZEN_PERSONAL_API_KEY"


def test_nameless_endpoint_keeps_its_host_keyed_slot():
    """Local servers are saved by URL alone; their slot must stay per host[:port]."""
    assert _custom_endpoint_key_identity("http://localhost:11434/v1", "") == "localhost_11434"
    assert _custom_endpoint_key_identity("http://172.17.0.1/v1", "  ") == "172.17.0.1"
    assert custom_endpoint_key_env(_custom_endpoint_key_identity(URL, "")) == custom_endpoint_key_env("opencode.ai")
