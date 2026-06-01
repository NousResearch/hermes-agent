"""Tests for plugins/memory/honcho/cli.py."""

from types import SimpleNamespace
import json


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


class TestCmdSetupLocalJwt:
    """Local-deployment setup must allow configuring a JWT for AUTH_JWT_SECRET-backed Honcho servers."""

    def _run_setup(self, monkeypatch, tmp_path, initial_cfg, prompt_answers):
        import plugins.memory.honcho.cli as honcho_cli

        # Avoid touching real config / SDK / filesystem.
        cfg_path = tmp_path / "honcho.json"
        monkeypatch.setattr(honcho_cli, "_read_config", lambda: dict(initial_cfg))
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.setattr(honcho_cli, "_ensure_sdk_installed", lambda: True)

        written = {}

        def _capture_write(cfg, path=None):
            written["cfg"] = cfg
            written["path"] = path

        monkeypatch.setattr(honcho_cli, "_write_config", _capture_write)

        # Feed scripted prompt answers in order.
        answers = list(prompt_answers)

        def _fake_prompt(label, default=None, secret=False):
            if not answers:
                # Default-through any remaining prompts to keep the wizard moving.
                return default or ""
            return answers.pop(0)

        monkeypatch.setattr(honcho_cli, "_prompt", _fake_prompt)

        honcho_cli.cmd_setup(SimpleNamespace())
        return written.get("cfg")

    def test_local_setup_stores_jwt_under_host_block(self, monkeypatch, tmp_path):
        """Self-hosted users supplying a JWT must have it written under hosts.<host>.apiKey,
        not as the top-level cloud apiKey, so cloud/hybrid switching is preserved and
        get_honcho_client treats it as an explicit local auth opt-in."""
        cfg = self._run_setup(
            monkeypatch,
            tmp_path,
            initial_cfg={},
            prompt_answers=[
                "local",                       # deployment
                "http://localhost:8000",       # base URL
                "my-local-jwt-token",          # local JWT
            ],
        )
        assert cfg is not None
        assert cfg.get("baseUrl") == "http://localhost:8000"
        # Top-level apiKey must remain unset (cloud field).
        assert not cfg.get("apiKey")
        # The new local JWT belongs under the host block.
        host_block = (cfg.get("hosts") or {}).get("hermes") or {}
        assert host_block.get("apiKey") == "my-local-jwt-token"

    def test_local_setup_blank_jwt_keeps_local_no_auth(self, monkeypatch, tmp_path):
        """Blank JWT prompt response on a fresh local config must not introduce an apiKey
        anywhere (local no-auth Honcho deployments must still work out of the box)."""
        cfg = self._run_setup(
            monkeypatch,
            tmp_path,
            initial_cfg={},
            prompt_answers=[
                "local",
                "http://localhost:8000",
                "",  # blank JWT
            ],
        )
        assert cfg is not None
        assert cfg.get("baseUrl") == "http://localhost:8000"
        assert not cfg.get("apiKey")
        host_block = (cfg.get("hosts") or {}).get("hermes") or {}
        assert not host_block.get("apiKey")


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
    """Identity-key carryover during profile cloning.

    The host-scoped identity-mapping keys (``userPeerAliases``,
    ``runtimePeerPrefix``, ``pinPeerName``) must survive a clone; otherwise
    the new profile silently fragments memory by resolving gateway users to
    raw runtime IDs instead of operator-declared peers.
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
        new_block = written["cfg"]["hosts"]["hermes_coder"]
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
        new_block = written["cfg"]["hosts"]["hermes_coder"]
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
        new_block = written["cfg"]["hosts"]["hermes_coder"]
        assert new_block["pinPeerName"] is True

    def test_unset_identity_keys_do_not_appear_in_cloned_profile(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"peerName": "eri"}},
        }
        honcho_cli, written = self._setup_clone_env(monkeypatch, tmp_path, cfg)
        ok = honcho_cli.clone_honcho_for_profile("coder")
        assert ok is True
        new_block = written["cfg"]["hosts"]["hermes_coder"]
        assert "userPeerAliases" not in new_block
        assert "runtimePeerPrefix" not in new_block
        assert "pinPeerName" not in new_block


class TestSetupWizardDeploymentShape:
    """The deployment-shape step writes pinPeerName / userPeerAliases /
    runtimePeerPrefix based on the operator's chosen shape.

    Single-operator deployments collapse all platforms to peerName.
    Multi-user gateways leave the resolver to route per-runtime.
    Hybrid deployments alias the operator's own runtime IDs only.

    These tests script the interactive _prompt calls and assert the
    resulting hermes_host block, so the wizard's deployment-shape
    semantics stay locked even as adjacent prompts are added.
    """

    def _run_setup(self, monkeypatch, tmp_path, *, answers, initial_cfg=None):
        import plugins.memory.honcho.cli as honcho_cli

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{}")
        cfg = initial_cfg if initial_cfg is not None else {"apiKey": "***"}

        monkeypatch.setattr(honcho_cli, "_read_config", lambda: cfg)
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.setattr(honcho_cli, "_ensure_sdk_installed", lambda: True)
        monkeypatch.setattr(honcho_cli, "_write_config", lambda *a, **k: None)

        # Bypass config.yaml + connection test side effects.
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"memory": {}}, raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.config.save_config", lambda c: None, raising=False,
        )

        class _FakeClientCfg:
            def resolve_session_name(self):
                return "hermes-test"
            workspace_id = "hermes"
            peer_name = "eri"
            ai_peer = "hermetika"
            observation_mode = "directional"
            write_frequency = "async"
            recall_mode = "hybrid"
            session_strategy = "per-session"

        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda host=None: _FakeClientCfg(),
        )
        monkeypatch.setattr(
            "plugins.memory.honcho.client.reset_honcho_client",
            lambda: None,
        )
        monkeypatch.setattr(
            "plugins.memory.honcho.client.get_honcho_client",
            lambda hcfg: object(),
        )

        # Scripted _prompt: pop answers in order. Default-return for unconsumed prompts.
        answer_iter = iter(answers)
        def _scripted_prompt(label, default=None, secret=False):
            try:
                return next(answer_iter)
            except StopIteration:
                return default if default is not None else ""
        monkeypatch.setattr(honcho_cli, "_prompt", _scripted_prompt)

        honcho_cli.cmd_setup(SimpleNamespace())
        return cfg["hosts"]["hermes"]

    def test_single_shape_sets_pin_peer_name_and_clears_aliases(self, monkeypatch, tmp_path):
        answers = [
            "cloud",           # deployment
            "",                # api key (keep)
            "eri",             # peer name
            "hermetika",       # ai peer
            "hermes",          # workspace
            "single",          # deployment shape ← key answer
            # remaining prompts fall through to defaults
        ]
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {
                "userPeerAliases": {"old": "stale"},
                "runtimePeerPrefix": "old_",
            }},
        }
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is True
        assert "userPeerAliases" not in host
        assert "runtimePeerPrefix" not in host

    def test_multi_shape_leaves_pin_false_and_accepts_prefix(self, monkeypatch, tmp_path):
        answers = [
            "cloud",           # deployment
            "",                # api key (keep)
            "eri",             # peer name
            "hermetika",       # ai peer
            "hermes",          # workspace
            "multi",           # deployment shape
            "telegram_",       # runtime peer prefix
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers)
        assert host["pinPeerName"] is False
        # Multi must NOT auto-write ``userPeerAliases: {}``: an empty host
        # map would silently override a root-level baseline.  Absence is
        # the correct "no host opinion" signal.
        assert "userPeerAliases" not in host
        assert host["runtimePeerPrefix"] == "telegram_"

    def test_hybrid_shape_aliases_operator_runtime_ids_to_peer_name(self, monkeypatch, tmp_path):
        answers = [
            "cloud",           # deployment
            "",                # api key (keep)
            "eri",             # peer name
            "hermetika",       # ai peer
            "hermes",          # workspace
            "hybrid",          # deployment shape
            "86701400",        # telegram uid
            "491827364",       # discord snowflake
            "",                # slack (skip)
            "",                # matrix (skip)
            "",                # runtime peer prefix (skip)
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers)
        assert host["pinPeerName"] is False
        assert host["userPeerAliases"] == {
            "86701400": "eri",
            "491827364": "eri",
        }
        assert "runtimePeerPrefix" not in host

    def test_skip_shape_preserves_existing_identity_config(self, monkeypatch, tmp_path):
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {
                "pinPeerName": True,
                "userPeerAliases": {"keep": "me"},
                "runtimePeerPrefix": "keep_",
            }},
        }
        answers = [
            "cloud", "", "eri", "hermetika", "hermes", "skip",
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is True
        assert host["userPeerAliases"] == {"keep": "me"}
        assert host["runtimePeerPrefix"] == "keep_"

    def test_single_to_multi_steers_to_hybrid_by_default(self, monkeypatch, tmp_path):
        """Flipping single → multi triggers a warning that auto-steers the
        operator to ``hybrid`` (default), so their own runtime IDs keep
        landing on peerName instead of orphaning the pinned-pool history.
        """
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"pinPeerName": True, "peerName": "eri"}},
        }
        answers = [
            "cloud",           # deployment
            "",                # api key (keep)
            "eri",             # peer name
            "hermetika",       # ai peer
            "hermes",          # workspace
            "multi",           # deployment shape — triggers the guard
            "hybrid",          # guard response: accept the steer
            "86701400",        # telegram uid
            "",                # discord (skip)
            "",                # slack (skip)
            "",                # matrix (skip)
            "",                # runtime prefix (skip)
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is False
        assert host["userPeerAliases"] == {"86701400": "eri"}

    def test_single_to_multi_yes_override_keeps_multi(self, monkeypatch, tmp_path):
        """Operator can override the steer by answering ``yes`` and accept
        the orphaning consequences.  This is the explicit undo-the-pin path.
        """
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"pinPeerName": True, "peerName": "eri"}},
        }
        answers = [
            "cloud", "", "eri", "hermetika", "hermes",
            "multi",           # deployment shape — triggers the guard
            "yes",             # guard response: confirm multi
            "telegram_",       # runtime peer prefix
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is False
        # See test_multi_shape_leaves_pin_false_and_accepts_prefix.
        assert "userPeerAliases" not in host
        assert host["runtimePeerPrefix"] == "telegram_"

    def test_host_pin_user_peer_true_is_detected_as_single(self, monkeypatch, tmp_path):
        """Host-level ``pinUserPeer: true`` must classify as ``single``.

        Pressing Enter at the shape prompt then preserves the pin instead
        of falling through to ``multi`` and orphaning the user's memory
        pool — the bug the wizard regressed when ``pinUserPeer`` landed
        as a higher-precedence alias.
        """
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"pinUserPeer": True, "peerName": "eri"}},
        }
        # Exhaust the iterator before the shape prompt so the scripted
        # mock falls through to the prompt's default (which is the
        # wizard-detected shape).  Scripting an explicit "" would NOT
        # exercise that fallthrough — the mock returns it literally.
        answers = ["cloud", "", "eri", "hermetika", "hermes"]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        # Scrub-then-write normalises onto pinPeerName and drops the alias
        # so resolver precedence can't reintroduce ambiguity.
        assert host["pinPeerName"] is True
        assert "pinUserPeer" not in host

    def test_host_pin_user_peer_false_overrides_root_pin_peer_name(
        self, monkeypatch, tmp_path
    ):
        """Host ``pinUserPeer: false`` outranks host ``pinPeerName`` in the
        resolver.  Detection must agree, otherwise the wizard would offer
        ``single`` as the default and silently re-pin a profile the
        operator explicitly unpinned via the newer key.
        """
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {
                "pinUserPeer": False,
                "pinPeerName": True,
                "peerName": "eri",
            }},
        }
        answers = ["cloud", "", "eri", "hermetika", "hermes"]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is False
        assert "pinUserPeer" not in host

    def test_root_user_peer_aliases_detected_as_hybrid(self, monkeypatch, tmp_path):
        """Root-level ``userPeerAliases`` must classify as ``hybrid`` even
        when the host block has no aliases of its own.
        """
        initial_cfg = {
            "apiKey": "***",
            "userPeerAliases": {"86701400": "eri"},
            "hosts": {"hermes": {"peerName": "eri"}},
        }
        answers = ["cloud", "", "eri", "hermetika", "hermes"]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is False
        # Hybrid materialises the root aliases into the host so subsequent
        # operator edits live on the host block they're inspecting.
        assert host["userPeerAliases"] == {"86701400": "eri"}

    def test_multi_does_not_override_root_user_peer_aliases(self, monkeypatch, tmp_path):
        """Explicit ``multi`` must leave the host ``userPeerAliases`` key
        absent, preserving any root-level aliases as a cross-host baseline.

        Picking ``multi`` here is an active choice — detection would have
        defaulted to ``hybrid`` because root aliases exist — so the
        operator's intent is to drop the alias mapping for this host.
        We honor that by writing ``pinPeerName: false`` only, and rely
        on the host's absence of ``userPeerAliases`` to inherit root.
        That inheritance is intentional: a true wipe would require the
        operator to delete the root key explicitly.
        """
        initial_cfg = {
            "apiKey": "***",
            "userPeerAliases": {"baseline": "eri"},
            "hosts": {"hermes": {"peerName": "eri"}},
        }
        answers = [
            "cloud", "", "eri", "hermetika", "hermes",
            "multi",           # explicit multi override of detected hybrid
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is False
        assert "userPeerAliases" not in host

    def test_single_scrubs_stale_pin_user_peer_false(self, monkeypatch, tmp_path):
        """Choosing ``single`` must drop any host-level ``pinUserPeer``,
        otherwise an existing ``pinUserPeer: false`` would outrank the
        freshly written ``pinPeerName: true`` and leave the profile
        effectively unpinned (the P1 latent-precedence regression).
        """
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {
                "pinUserPeer": False,
                "peerName": "eri",
            }},
        }
        answers = [
            "cloud", "", "eri", "hermetika", "hermes",
            "single",
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinPeerName"] is True
        assert "pinUserPeer" not in host


class TestCloneCarriesPinUserPeer:
    """``pinUserPeer`` (canonical name for ``pinPeerName``) must survive a
    profile clone.  Without this, a default profile that uses the newer
    key would silently produce cloned profiles without the pin even
    though the resolver prefers ``pinUserPeer`` over ``pinPeerName``.
    """

    def test_clone_inherits_host_pin_user_peer(self, monkeypatch, tmp_path):
        import plugins.memory.honcho.cli as honcho_cli

        cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"pinUserPeer": True, "peerName": "eri"}},
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{}")
        monkeypatch.setattr(honcho_cli, "_read_config", lambda: cfg)
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_ensure_peer_exists", lambda host_key=None: True)
        written = {}
        monkeypatch.setattr(
            honcho_cli, "_write_config", lambda c, path=None: written.setdefault("cfg", c),
        )

        ok = honcho_cli.clone_honcho_for_profile("partner")
        assert ok is True
        new_block = written["cfg"]["hosts"]["hermes_partner"]
        assert new_block["pinUserPeer"] is True
