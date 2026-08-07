"""Regression tests for transport mutual-exclusivity when merging MCP server
configs via ``_deep_merge``.

See issue #78606: a profile config that switches a server from stdio
(``command``/``args``) to HTTP (``url``) must not leave the stdio fields in the
merged config, otherwise the server ends up with *both* transports and the
loader registers duplicate tool instances.
"""

from hermes_cli.config import _deep_merge


class TestMcpServerTransportExclusivity:
    """Layer 1-4: per-server transport switching during deep merge."""

    def test_http_override_strips_stdio_fields(self):
        """L1: override with ``url`` (no ``command``) drops stdio fields."""
        base = {
            "mcp_servers": {
                "rca": {
                    "command": "/usr/local/bin/node",
                    "args": ["/path/to/server/index.js"],
                    "env": {"NODE_ENV": "production"},
                    "enabled": True,
                }
            }
        }
        override = {
            "mcp_servers": {
                "rca": {
                    "url": "http://localhost:3001/mcp",
                    "enabled": True,
                }
            }
        }
        merged = _deep_merge(base, override)
        server = merged["mcp_servers"]["rca"]
        assert "url" in server
        assert server["url"] == "http://localhost:3001/mcp"
        # stdio-only fields must be gone
        assert "command" not in server
        assert "args" not in server
        assert "env" not in server

    def test_stdio_override_strips_http_fields(self):
        """L2: override with ``command`` (no ``url``) drops HTTP fields."""
        base = {
            "mcp_servers": {
                "rca": {
                    "url": "http://localhost:3001/mcp",
                    "headers": {"Authorization": "Bearer x"},
                    "enabled": True,
                }
            }
        }
        override = {
            "mcp_servers": {
                "rca": {
                    "command": "/usr/local/bin/node",
                    "args": ["/path/to/server/index.js"],
                    "enabled": True,
                }
            }
        }
        merged = _deep_merge(base, override)
        server = merged["mcp_servers"]["rca"]
        assert "command" in server
        assert server["command"] == "/usr/local/bin/node"
        # HTTP-only fields must be gone
        assert "url" not in server
        assert "headers" not in server

    def test_non_transport_fields_survive_transport_switch(self):
        """L3: switching transport keeps non-transport fields from both."""
        base = {
            "mcp_servers": {
                "rca": {
                    "command": "/usr/local/bin/node",
                    "args": ["index.js"],
                    "description": "RCA stdio server",
                    "enabled": True,
                }
            }
        }
        override = {
            "mcp_servers": {
                "rca": {
                    "url": "http://localhost:3001/mcp",
                    "enabled": False,  # override disables it
                }
            }
        }
        merged = _deep_merge(base, override)
        server = merged["mcp_servers"]["rca"]
        assert "url" in server
        assert "command" not in server
        # override's enabled wins
        assert server["enabled"] is False
        # base-only non-transport field survives
        assert server["description"] == "RCA stdio server"

    def test_same_transport_merge_unchanged(self):
        """L4: both stdio — normal field-level recursion still applies."""
        base = {
            "mcp_servers": {
                "rca": {
                    "command": "/usr/local/bin/node",
                    "args": ["old.js"],
                    "enabled": True,
                }
            }
        }
        override = {
            "mcp_servers": {
                "rca": {
                    "command": "/usr/local/bin/node",
                    "args": ["new.js"],  # change args, keep command
                }
            }
        }
        merged = _deep_merge(base, override)
        server = merged["mcp_servers"]["rca"]
        assert server["command"] == "/usr/local/bin/node"
        assert server["args"] == ["new.js"]
        assert server["enabled"] is True

    def test_both_http_same_transport_merge_unchanged(self):
        """L4 (mirror): both HTTP — normal field-level recursion applies."""
        base = {
            "mcp_servers": {
                "rca": {
                    "url": "http://localhost:3001/mcp",
                    "headers": {"X-Old": "1"},
                    "enabled": True,
                }
            }
        }
        override = {
            "mcp_servers": {
                "rca": {
                    "url": "http://localhost:3002/mcp",  # new url
                    "headers": {"X-New": "2"},
                }
            }
        }
        merged = _deep_merge(base, override)
        server = merged["mcp_servers"]["rca"]
        assert server["url"] == "http://localhost:3002/mcp"
        assert server["enabled"] is True

    def test_new_server_added_normally(self):
        """A server present only in the override is added as-is."""
        base = {"mcp_servers": {"a": {"command": "x", "enabled": True}}}
        override = {"mcp_servers": {"b": {"url": "http://x", "enabled": True}}}
        merged = _deep_merge(base, override)
        assert "a" in merged["mcp_servers"]
        assert "b" in merged["mcp_servers"]
        assert merged["mcp_servers"]["b"]["url"] == "http://x"


class TestDeepMergeNotRegressed:
    """Layer 5: generic merge behavior is unchanged for other keys."""

    def test_generic_nested_merge_preserved(self):
        """Non-mcp_servers keys still deep-merge recursively."""
        base = {"tts": {"elevenlabs": {"voice_id": "v1", "model_id": "m1"}}}
        override = {"tts": {"elevenlabs": {"voice_id": "v2"}}}
        merged = _deep_merge(base, override)
        assert merged["tts"]["elevenlabs"]["voice_id"] == "v2"
        assert merged["tts"]["elevenlabs"]["model_id"] == "m1"

    def test_generic_top_level_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_non_mcp_dict_named_mcp_servers_mirrors_is_handled(self):
        """A top-level key literally named mcp_servers gets the special path."""
        # Sanity: the special-casing keys on the string 'mcp_servers'.
        base = {"mcp_servers": {"s": {"command": "c", "args": ["a"], "env": {"E": "1"}}}}
        override = {"mcp_servers": {"s": {"url": "http://u"}}}
        merged = _deep_merge(base, override)
        assert "command" not in merged["mcp_servers"]["s"]
        assert merged["mcp_servers"]["s"]["url"] == "http://u"
