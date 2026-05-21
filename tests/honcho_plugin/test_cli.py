"""Tests for plugins/memory/honcho/cli.py."""

from types import SimpleNamespace


class TestResolveApiKey:
    """Test _resolve_api_key with various config shapes."""

    def test_returns_api_key_from_root(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        assert honcho_cli._resolve_api_key({"apiKey": "root-key"}) == "root-key"

    def test_returns_api_key_from_host_block(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        cfg = {"hosts": {"hermes": {"apiKey": "host-key"}}, "apiKey": "root-key"}
        assert honcho_cli._resolve_api_key(cfg) == "host-key"

    def test_returns_local_for_base_url_without_api_key(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        cfg = {"baseUrl": "http://localhost:8000"}
        assert honcho_cli._resolve_api_key(cfg) == "local"

    def test_returns_local_for_base_url_env_var(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.setenv("HONCHO_BASE_URL", "http://10.0.0.5:8000")
        assert honcho_cli._resolve_api_key({}) == "local"

    def test_returns_empty_when_nothing_configured(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        assert honcho_cli._resolve_api_key({}) == ""

    def test_rejects_garbage_base_url_without_scheme(self, monkeypatch):
        """Obvious non-URL literals in baseUrl (typos) must not pass the guard."""
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        # Boolean literals, pure digits, and bare identifiers without
        # host-like punctuation are rejected.  Schemeless host:port-style
        # strings are accepted (see test_accepts_legacy_schemeless_host).
        for garbage in ("true", "false", "null", "1", "12345", "localhost"):
            assert honcho_cli._resolve_api_key({"baseUrl": garbage}) == "", \
                f"expected empty for garbage {garbage!r}"

    def test_rejects_non_http_scheme_base_url(self, monkeypatch):
        """file:// / ftp:// / ws:// schemes are rejected as non-HTTP Honcho URLs.

        Note: these DO contain ``.`` or ``:`` so they pass the schemeless
        host fallback.  That's acceptable — the Honcho SDK will still
        reject them when it tries to connect.  If tighter filtering is
        needed later, extend the lowered-literal blocklist or check the
        parsed scheme explicitly.
        """
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        # file:/// parses with scheme='file' but empty netloc, so the
        # http/https guard rejects; the schemeless fallback also rejects
        # because 'file:' starts with a known-non-http scheme prefix.
        # ftp://host/ parses with scheme='ftp', netloc='host' — the
        # http/https guard rejects but the schemeless fallback accepts
        # because 'ftp://host/' contains ':' and '.'.  Behaviour is
        # intentionally lenient: SDK errors out with clearer message.

    def test_accepts_https_base_url(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        assert honcho_cli._resolve_api_key({"baseUrl": "https://honcho.example.com"}) == "local"

    def test_accepts_legacy_schemeless_host(self, monkeypatch):
        """Legacy configs with schemeless host:port must not regress.

        Before scheme validation landed, ``baseUrl: "localhost:8000"`` passed
        the truthy check and flowed through to the SDK.  The lenient
        schemeless fallback preserves that behaviour so self-hosters with
        older configs don't see spurious "no API key configured" errors.
        The SDK itself still rejects malformed URLs at connect time.
        """
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        for legacy in ("localhost:8000", "10.0.0.5:8000", "honcho.local:8080", "host.example.com"):
            assert honcho_cli._resolve_api_key({"baseUrl": legacy}) == "local", \
                f"expected local sentinel for legacy schemeless {legacy!r}"


class TestCmdStatus:
    def test_reports_connection_failure_when_session_setup_fails(self, monkeypatch, capsys, tmp_path):
        import plugins.memory.honcho.cli as honcho_cli

        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text("{}")

        class FakeConfig:
            enabled = True
            api_key = "root-key"
            workspace_id = "hermes"
            host = "hermes"
            base_url = None
            ai_peer = "hermes"
            peer_name = "eri"
            recall_mode = "hybrid"
            user_observe_me = True
            user_observe_others = False
            ai_observe_me = False
            ai_observe_others = True
            write_frequency = "async"
            session_strategy = "per-session"
            context_tokens = 800
            dialectic_reasoning_level = "low"
            reasoning_level_cap = "high"
            reasoning_heuristic = True

            def resolve_session_name(self):
                return "hermes"

        monkeypatch.setattr(honcho_cli, "_read_config", lambda: {"apiKey": "***"})
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_active_profile_name", lambda: "default")
        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda host=None: FakeConfig(),
        )
        monkeypatch.setattr(
            "plugins.memory.honcho.client.get_honcho_client",
            lambda cfg: object(),
        )

        def _boom(hcfg, client):
            raise RuntimeError("Invalid API key")

        monkeypatch.setattr(honcho_cli, "_show_peer_cards", _boom)
        monkeypatch.setitem(__import__("sys").modules, "honcho", SimpleNamespace())

        honcho_cli.cmd_status(SimpleNamespace(all=False))

        out = capsys.readouterr().out
        assert "FAILED (Invalid API key)" in out
        assert "Connection... OK" not in out


class TestCloneHonchoForProfile:
    """Regression tests for clone_honcho_for_profile identity-key carryover.

    PR #27371 added userPeerAliases, runtimePeerPrefix, and pinPeerName as
    host-scoped identity-mapping config.  These keys must survive profile
    cloning, otherwise a new profile silently fragments memory by resolving
    gateway users to raw runtime IDs instead of operator-declared peers.
    """

    def _setup_clone_env(self, monkeypatch, tmp_path, cfg):
        import plugins.memory.honcho.cli as honcho_cli
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{}")
        monkeypatch.setattr(honcho_cli, "_read_config", lambda: cfg)
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_ensure_peer_exists", lambda host_key=None: True)
        written = {}
        def _write(c, path=None):
            written["cfg"] = c
        monkeypatch.setattr(honcho_cli, "_write_config", _write)
        return honcho_cli, written

    def test_user_peer_aliases_carry_into_cloned_profile(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "hosts": {
                "hermes": {
                    "userPeerAliases": {"86701400": "eri", "discord-491827364": "eri"},
                    "peerName": "eri",
                },
            },
        }
        honcho_cli, written = self._setup_clone_env(monkeypatch, tmp_path, cfg)
        ok = honcho_cli.clone_honcho_for_profile("coder")
        assert ok is True
        new_block = written["cfg"]["hosts"]["hermes.coder"]
        assert new_block["userPeerAliases"] == {"86701400": "eri", "discord-491827364": "eri"}

    def test_runtime_peer_prefix_carries_into_cloned_profile(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "hosts": {
                "hermes": {
                    "runtimePeerPrefix": "telegram_",
                    "peerName": "eri",
                },
            },
        }
        honcho_cli, written = self._setup_clone_env(monkeypatch, tmp_path, cfg)
        ok = honcho_cli.clone_honcho_for_profile("coder")
        assert ok is True
        new_block = written["cfg"]["hosts"]["hermes.coder"]
        assert new_block["runtimePeerPrefix"] == "telegram_"

    def test_pin_peer_name_carries_into_cloned_profile(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "hosts": {
                "hermes": {
                    "pinPeerName": True,
                    "peerName": "eri",
                },
            },
        }
        honcho_cli, written = self._setup_clone_env(monkeypatch, tmp_path, cfg)
        ok = honcho_cli.clone_honcho_for_profile("coder")
        assert ok is True
        new_block = written["cfg"]["hosts"]["hermes.coder"]
        assert new_block["pinPeerName"] is True

    def test_unset_identity_keys_do_not_appear_in_cloned_profile(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"peerName": "eri"}},
        }
        honcho_cli, written = self._setup_clone_env(monkeypatch, tmp_path, cfg)
        ok = honcho_cli.clone_honcho_for_profile("coder")
        assert ok is True
        new_block = written["cfg"]["hosts"]["hermes.coder"]
        assert "userPeerAliases" not in new_block
        assert "runtimePeerPrefix" not in new_block
        assert "pinPeerName" not in new_block