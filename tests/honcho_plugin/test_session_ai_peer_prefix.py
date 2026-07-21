"""Tests for the ``sessionAiPeerPrefix`` config flag.

``sessionPeerPrefix`` prefixes a resolved session name with the *user* peer
(``peerName``).  ``sessionAiPeerPrefix`` is the symmetric counterpart for the
*AI* peer (``aiPeer``): when several AI peers share one workspace + peerName +
gateway chat key, the ``gateway_session_key`` branch of
``resolve_session_name`` returns an AI-peer-agnostic name and every peer
collides on the same Honcho session.  Setting ``sessionAiPeerPrefix: true``
prefixes the final name with ``{ai_peer}-`` so the sessions stay disjoint.

Tests cover config parsing (``client.py::from_global_config``) and the public
resolver wrapper across every resolution path, including the length cap.
"""

import json

from plugins.memory.honcho.client import HonchoClientConfig


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestSessionAiPeerPrefixConfigParsing:
    def test_default_is_false(self):
        """Default preserves existing behaviour."""
        config = HonchoClientConfig()
        assert config.session_ai_peer_prefix is False

    def test_root_level_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "eli",
            "aiPeer": "ivy",
            "sessionAiPeerPrefix": True,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.session_ai_peer_prefix is True
        assert config.ai_peer == "ivy"

    def test_host_block_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "eli",
            "hosts": {
                "hermes": {"aiPeer": "ivy", "sessionAiPeerPrefix": True},
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.session_ai_peer_prefix is True

    def test_host_block_overrides_root(self, tmp_path, monkeypatch):
        """Host block wins over root — matches every other flag."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "eli",
            "sessionAiPeerPrefix": True,
            "hosts": {
                "hermes": {"sessionAiPeerPrefix": False},
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.session_ai_peer_prefix is False


# ---------------------------------------------------------------------------
# Resolver behaviour
# ---------------------------------------------------------------------------


class TestSessionAiPeerPrefixResolver:
    def _config(self, **overrides):
        base = dict(
            ai_peer="ivy",
            peer_name="eli",
            session_ai_peer_prefix=True,
        )
        base.update(overrides)
        return HonchoClientConfig(**base)

    def test_gateway_key_gets_ai_peer_prefix(self):
        """The motivating case: gateway_session_key is AI-peer-agnostic."""
        cfg = self._config()
        name = cfg.resolve_session_name(
            gateway_session_key="agent:main:telegram:dm:8055661863",
        )
        assert name.startswith("ivy-")
        assert "telegram-dm-8055661863" in name

    def test_disabled_leaves_name_untouched(self):
        """Regression guard — default behaviour is unchanged."""
        cfg = self._config(session_ai_peer_prefix=False)
        name = cfg.resolve_session_name(
            gateway_session_key="agent:main:telegram:dm:8055661863",
        )
        assert not name.startswith("ivy-")

    def test_two_ai_peers_stay_disjoint(self):
        """Same workspace + chat key, different aiPeer -> different sessions."""
        key = "agent:main:telegram:dm:8055661863"
        ivy = self._config(ai_peer="ivy").resolve_session_name(gateway_session_key=key)
        holly = self._config(ai_peer="holly").resolve_session_name(gateway_session_key=key)
        assert ivy != holly
        assert ivy.startswith("ivy-")
        assert holly.startswith("holly-")

    def test_applies_across_strategy_paths(self, tmp_path):
        """Prefix is not limited to the gateway-key branch."""
        cfg = self._config(session_strategy="per-directory")
        name = cfg.resolve_session_name(cwd=str(tmp_path / "myproject"))
        assert name == "ivy-myproject"

    def test_no_ai_peer_no_prefix(self):
        """An empty aiPeer must not produce a stray leading hyphen."""
        cfg = self._config(ai_peer="")
        name = cfg.resolve_session_name(gateway_session_key="agent:main:dm:1")
        assert not name.startswith("-")

    def test_prefix_respects_session_id_length_cap(self):
        """A long base + prefix stays within Honcho's 100-char session-id cap,
        and remains disjoint per AI peer (hash incorporates the prefix)."""
        long_key = "agent:main:telegram:dm:" + "9" * 200
        ivy = self._config(ai_peer="ivy").resolve_session_name(gateway_session_key=long_key)
        holly = self._config(ai_peer="holly").resolve_session_name(gateway_session_key=long_key)
        assert len(ivy) <= HonchoClientConfig._HONCHO_SESSION_ID_MAX_LEN
        assert len(holly) <= HonchoClientConfig._HONCHO_SESSION_ID_MAX_LEN
        assert ivy != holly
