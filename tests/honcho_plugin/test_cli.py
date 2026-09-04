"""Tests for plugins/memory/honcho/cli.py."""

from types import SimpleNamespace
import json

import pytest


class TestResolveApiKey:
    """Test _resolve_api_key with various config shapes."""

    def test_returns_api_key_from_root(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        assert honcho_cli._resolve_api_key({"apiKey": "root-key"}) == "root-key"


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

    def test_auth_line_detects_oauth_grant(self, monkeypatch, capsys, tmp_path):
        import plugins.memory.honcho.cli as honcho_cli

        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text("{}")

        class FakeConfig:
            enabled = True
            api_key = "hch-at-deadbeef"
            workspace_id = "claude-code"
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
            context_tokens = None
            dialectic_reasoning_level = "low"
            reasoning_level_cap = "high"
            reasoning_heuristic = True
            raw = {
                "hosts": {
                    "hermes": {
                        "apiKey": "hch-at-deadbeef",
                        "oauth": {
                            "refreshToken": "hch-rt-x",
                            "clientId": "hermes-agent",
                            "tokenEndpoint": "https://api.honcho.dev/oauth/token",
                            "expiresAt": 9999999999,
                        },
                    }
                }
            }

            def resolve_session_name(self):
                return "hermes"

        monkeypatch.setattr(honcho_cli, "_read_config", lambda: {})
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_active_profile_name", lambda: "default")
        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda host=None: FakeConfig(),
        )
        monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", lambda cfg: object())
        monkeypatch.setattr(honcho_cli, "_show_peer_cards", lambda hcfg, client: None)
        monkeypatch.setitem(__import__("sys").modules, "honcho", SimpleNamespace())

        honcho_cli.cmd_status(SimpleNamespace(all=False))

        out = capsys.readouterr().out
        assert "Auth:           OAuth (hermes-agent" in out
        assert "API key:" not in out


class TestCloneHonchoForProfile:
    """Identity-key carryover during profile cloning.

    The host-scoped identity-mapping keys (``userPeerAliases``,
    ``runtimePeerPrefix``, ``pinUserPeer``) must survive a clone; otherwise
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
                    "userPeerAliases": {"7654321": "eri", "discord-491827364": "eri"},
                    "peerName": "eri",
                },
            },
        }
        honcho_cli, written = self._setup_clone_env(monkeypatch, tmp_path, cfg)
        ok = honcho_cli.clone_honcho_for_profile("coder")
        assert ok is True
        new_block = written["cfg"]["hosts"]["hermes_coder"]
        assert new_block["userPeerAliases"] == {"7654321": "eri", "discord-491827364": "eri"}

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

    def test_legacy_pin_peer_name_migrates_to_canonical_on_clone(self, monkeypatch, tmp_path):
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
        assert new_block["pinUserPeer"] is True
        assert "pinPeerName" not in new_block

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
        assert "pinUserPeer" not in new_block
        assert "pinPeerName" not in new_block


class TestSetupWizardDeploymentShape:
    """The gateway identity-mapping tree writes pinUserPeer / userPeerAliases /
    runtimePeerPrefix based on the operator's intent.

    Choice [1] (just me) collapses all platforms to peerName.
    Choice [3] (only other people) leaves the resolver to route per-runtime.
    Choice [2] (me + others, pooled) aliases the operator's own runtime IDs.

    These tests mock gateway detection and script the interactive _prompt
    calls, asserting the resulting hermes_host block so the tree's routing
    semantics stay locked even as adjacent prompts are added.
    """

    def _run_setup(self, monkeypatch, tmp_path, *, answers, initial_cfg=None,
                   gateway_platforms=("telegram",)):
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
        # No network probe / environment sniffing in tests.
        monkeypatch.setattr(honcho_cli, "_device_login_available", lambda: False)
        monkeypatch.setattr(honcho_cli, "_headless", lambda: (False, True))
        # Gate detection is mocked so tests control whether the tree runs.
        # None → undetectable; list (possibly empty) → connected platforms.
        gw = None if gateway_platforms is None else list(gateway_platforms)
        monkeypatch.setattr(honcho_cli, "_gateway_platforms", lambda: gw)

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
            # Auth-method prompt is orthogonal to shape; auto-answer apikey so the answer lists stay shape-only.
            if "OAuth" in label:
                return "apikey"
            try:
                return next(answer_iter)
            except StopIteration:
                return default if default is not None else ""
        monkeypatch.setattr(honcho_cli, "_prompt", _scripted_prompt)

        honcho_cli.cmd_setup(SimpleNamespace())
        return cfg["hosts"]["hermes"]

    def test_just_me_pins_and_clears_aliases(self, monkeypatch, tmp_path):
        answers = [
            "cloud",           # deployment
            "",                # api key (keep)
            "eri",             # peer name
            "hermetika",       # ai peer
            "hermes",          # workspace
            "1",               # tree: just me ← key answer
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
        assert host["pinUserPeer"] is True
        assert "userPeerAliases" not in host
        assert "runtimePeerPrefix" not in host

    def test_only_others_leaves_pin_false_and_accepts_prefix(self, monkeypatch, tmp_path):
        answers = [
            "cloud",           # deployment
            "",                # api key (keep)
            "eri",             # peer name
            "hermetika",       # ai peer
            "hermes",          # workspace
            "3",               # tree: only other people
            "telegram_",       # runtime peer prefix
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers)
        assert host["pinUserPeer"] is False
        # Multi must NOT auto-write ``userPeerAliases: {}``: an empty host
        # map would silently override a root-level baseline.  Absence is
        # the correct "no host opinion" signal.
        assert "userPeerAliases" not in host
        assert host["runtimePeerPrefix"] == "telegram_"

    def test_pooled_aliases_operator_runtime_ids_to_peer_name(self, monkeypatch, tmp_path):
        answers = [
            "cloud",           # deployment
            "",                # api key (keep)
            "eri",             # peer name
            "hermetika",       # ai peer
            "hermes",          # workspace
            "2",               # tree: me + other people
            "y",               # keep my memory pooled? → hybrid
            "7654321",        # telegram uid
            "491827364",       # discord snowflake
            "",                # slack (skip)
            "",                # matrix (skip)
            "",                # runtime peer prefix (skip)
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers)
        assert host["pinUserPeer"] is False
        assert host["userPeerAliases"] == {
            "7654321": "eri",
            "491827364": "eri",
        }
        assert "runtimePeerPrefix" not in host

    def test_skip_shape_preserves_existing_identity_config(self, monkeypatch, tmp_path):
        # Seeds the legacy ``pinPeerName``: skip must leave the mapping intact
        # except for the on-load migration onto the canonical key.
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {
                "pinPeerName": True,
                "userPeerAliases": {"keep": "me"},
                "runtimePeerPrefix": "keep_",
            }},
        }
        answers = [
            "cloud", "", "eri", "hermetika", "hermes", "s",
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinUserPeer"] is True
        assert "pinPeerName" not in host
        assert host["userPeerAliases"] == {"keep": "me"}
        assert host["runtimePeerPrefix"] == "keep_"

    def test_unpin_steers_to_pooled_by_default(self, monkeypatch, tmp_path):
        """Choosing 'only other people' on a currently-pinned profile triggers
        the orphan warning, which auto-steers to pooled (hybrid) so the
        operator's own runtime IDs keep landing on peerName.
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
            "3",               # tree: only others — triggers the orphan guard
            "y",               # pool my own memory instead? → hybrid
            "7654321",        # telegram uid
            "",                # discord (skip)
            "",                # slack (skip)
            "",                # matrix (skip)
            "",                # runtime prefix (skip)
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinUserPeer"] is False
        assert host["userPeerAliases"] == {"7654321": "eri"}


    def test_host_pin_user_peer_true_is_detected_as_single(self, monkeypatch, tmp_path):
        """Host-level ``pinUserPeer: true`` must classify as ``single``.

        Pressing Enter at the choice prompt then preserves the pin instead
        of falling through to per-user routing and orphaning the user's
        memory pool — the bug the wizard regressed when ``pinUserPeer``
        landed as a higher-precedence alias.
        """
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"pinUserPeer": True, "peerName": "eri"}},
        }
        # Exhaust the iterator before the choice prompt so the scripted
        # mock falls through to the prompt's default (the detected shape →
        # choice "1").  Scripting an explicit "" would NOT exercise that
        # fallthrough — the mock returns it literally.
        answers = ["cloud", "", "eri", "hermetika", "hermes"]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        # Scrub-then-write normalises onto the canonical pinUserPeer.
        assert host["pinUserPeer"] is True
        assert "pinPeerName" not in host


    def test_root_user_peer_aliases_detected_as_hybrid(self, monkeypatch, tmp_path):
        """Root-level ``userPeerAliases`` must classify as ``hybrid`` even
        when the host block has no aliases of its own.
        """
        initial_cfg = {
            "apiKey": "***",
            "userPeerAliases": {"7654321": "eri"},
            "hosts": {"hermes": {"peerName": "eri"}},
        }
        answers = ["cloud", "", "eri", "hermetika", "hermes"]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg)
        assert host["pinUserPeer"] is False
        # Hybrid materialises the root aliases into the host so subsequent
        # operator edits live on the host block they're inspecting.
        assert host["userPeerAliases"] == {"7654321": "eri"}


    def test_no_gateway_connected_skips_mapping_when_declined(self, monkeypatch, tmp_path):
        """With no gateway platforms connected, the tree is gated off; declining
        the 'configure anyway?' prompt leaves identity mapping untouched."""
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"peerName": "eri"}},
        }
        answers = ["cloud", "", "eri", "hermetika", "hermes", "n"]
        host = self._run_setup(
            monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg,
            gateway_platforms=[],
        )
        assert "pinUserPeer" not in host
        assert "userPeerAliases" not in host
        assert "runtimePeerPrefix" not in host

    def test_undetectable_gateway_skips_mapping_when_declined(self, monkeypatch, tmp_path):
        """When the gateway package can't be inspected (None), the wizard asks
        whether the gateway is running; 'no' skips the mapping step."""
        initial_cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"peerName": "eri"}},
        }
        answers = ["cloud", "", "eri", "hermetika", "hermes", "n"]
        host = self._run_setup(
            monkeypatch, tmp_path, answers=answers, initial_cfg=initial_cfg,
            gateway_platforms=None,
        )
        assert "pinUserPeer" not in host

    def test_raw_edit_sets_resolver_knobs_directly(self, monkeypatch, tmp_path):
        """The [e] escape hatch lets a power user set pinUserPeer + an alias +
        prefix directly, bypassing the intent tree."""
        answers = [
            "cloud", "", "eri", "hermetika", "hermes",
            "e",               # tree: edit raw keys
            "false",           # pinUserPeer
            "99887766=eri",    # one alias pair
            "",                # finish aliases
            "discord_",        # runtimePeerPrefix
        ]
        host = self._run_setup(monkeypatch, tmp_path, answers=answers)
        assert host["pinUserPeer"] is False
        assert host["userPeerAliases"] == {"99887766": "eri"}
        assert host["runtimePeerPrefix"] == "discord_"


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


class TestMigratePinKey:
    """``_migrate_pin_key`` rewrites the legacy ``pinPeerName`` onto the
    canonical ``pinUserPeer`` in place, without clobbering an existing
    canonical value."""


    def test_canonical_key_wins_when_both_present(self):
        import plugins.memory.honcho.cli as honcho_cli
        block = {"pinPeerName": True, "pinUserPeer": False}
        assert honcho_cli._migrate_pin_key(block) is True
        assert block == {"pinUserPeer": False}

    def test_noop_when_no_legacy_key(self):
        import plugins.memory.honcho.cli as honcho_cli
        block = {"pinUserPeer": True}
        assert honcho_cli._migrate_pin_key(block) is False
        assert block == {"pinUserPeer": True}


class TestCmdSetupDeviceFlow:
    """The cloud auth-method menu's device-code branch (RFC 8628)."""

    def _run_setup(self, monkeypatch, tmp_path, *, answers, device_available=True,
                   headless=(False, True), device_result=None, device_error=None):
        """Run cmd_setup with the device flow stubbed; returns (cfg, calls, prompts)."""
        import plugins.memory.honcho.cli as honcho_cli
        import plugins.memory.honcho.oauth_flow as oauth_flow
        from plugins.memory.honcho.oauth import OAuthCredential

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{}")
        cfg = {"apiKey": "***"}

        monkeypatch.setattr(honcho_cli, "_read_config", lambda: cfg)
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.setattr(honcho_cli, "_ensure_sdk_installed", lambda: True)
        monkeypatch.setattr(honcho_cli, "_write_config", lambda *a, **k: None)
        monkeypatch.setattr(honcho_cli, "_gateway_platforms", lambda: [])
        monkeypatch.setattr(honcho_cli, "_device_login_available", lambda: device_available)
        monkeypatch.setattr(honcho_cli, "_headless", lambda: headless)
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
        monkeypatch.setattr("plugins.memory.honcho.client.reset_honcho_client", lambda: None)
        monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", lambda hcfg: object())

        calls: list[dict] = []
        cred = OAuthCredential(
            access_token="hch-at-x", refresh_token="hch-rt-x", expires_at=9_999_999_999,
            client_id="hermes-agent", token_endpoint="http://x/oauth/token",
            consent_peer_name="lyra",
        )

        def fake_device_flow(**kwargs):
            calls.append(kwargs)
            if device_error is not None:
                raise device_error
            return device_result or cred

        monkeypatch.setattr(oauth_flow, "authorize_via_device_code", fake_device_flow)

        prompts: list[tuple[str, str | None]] = []
        answer_iter = iter(answers)
        def _scripted_prompt(label, default=None, secret=False):
            prompts.append((label, default))
            try:
                # Mirror the real _prompt: blank input falls back to the default.
                return next(answer_iter) or (default or "")
            except StopIteration:
                return default if default is not None else ""
        monkeypatch.setattr(honcho_cli, "_prompt", _scripted_prompt)

        honcho_cli.cmd_setup(SimpleNamespace())
        return cfg, calls, prompts

    def test_device_choice_runs_flow_and_stores_grant(self, monkeypatch, tmp_path):
        cfg, calls, _ = self._run_setup(monkeypatch, tmp_path, answers=["cloud", "device"])
        assert len(calls) == 1
        assert calls[0]["apply_config"] is False
        assert calls[0]["source"] == "hermes-cli"
        host = cfg["hosts"]["hermes"]
        assert host["apiKey"] == "hch-at-x"
        assert host["oauth"]["refreshToken"] == "hch-rt-x"
        assert host["peerName"] == "lyra"

    def test_headless_defaults_to_device(self, monkeypatch, tmp_path):
        # Blank answer takes the prompt default, which flips to device on a
        # remote/no-browser environment.
        cfg, calls, prompts = self._run_setup(
            monkeypatch, tmp_path, answers=["cloud", ""], headless=(True, False),
        )
        method_prompts = [p for p in prompts if "apikey" in p[0]]
        assert method_prompts[0][1] == "device"
        assert len(calls) == 1
        assert calls[0]["open_url"] is None  # never auto-open a browser headless
        assert cfg["hosts"]["hermes"]["apiKey"] == "hch-at-x"

    def test_denied_device_flow_aborts_without_grant(self, monkeypatch, tmp_path):
        from plugins.memory.honcho.oauth_flow import AccessDenied

        cfg, calls, _ = self._run_setup(
            monkeypatch, tmp_path, answers=["cloud", "device"],
            device_error=AccessDenied("access_denied", "user denied"),
        )
        assert len(calls) == 1
        assert "apiKey" not in cfg.get("hosts", {}).get("hermes", {})



class TestWriteRefusesUnparseableStore:
    """A honcho.json that exists but does not parse reads as {} on the tolerant path. Writing that
    back would replace every host and root key with the current command's block, so writes refuse."""

    def _point_at(self, monkeypatch, cfg_path):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes_coder")
        monkeypatch.setattr(honcho_cli, "_ensure_peer_exists", lambda host_key=None: False)
        return honcho_cli

    def test_write_config_refuses_and_leaves_file_alone(self, monkeypatch, tmp_path):
        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text('{"hosts": {"hermes": {"apiKey": "k"', encoding="utf-8")
        honcho_cli = self._point_at(monkeypatch, cfg_path)
        with pytest.raises(honcho_cli.ConfigWriteRefused):
            honcho_cli._write_config({"hosts": {"hermes_coder": {"enabled": True}}})
        assert cfg_path.read_text(encoding="utf-8") == '{"hosts": {"hermes": {"apiKey": "k"'

    def test_write_config_still_bootstraps_a_missing_file(self, monkeypatch, tmp_path):
        cfg_path = tmp_path / "honcho.json"
        honcho_cli = self._point_at(monkeypatch, cfg_path)
        honcho_cli._write_config({"apiKey": "k"})
        assert json.loads(cfg_path.read_text(encoding="utf-8")) == {"apiKey": "k"}

    def test_command_prints_one_sentence_and_writes_nothing(self, monkeypatch, tmp_path, capsys):
        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text("{not json", encoding="utf-8")
        honcho_cli = self._point_at(monkeypatch, cfg_path)
        honcho_cli.honcho_command(SimpleNamespace(honcho_command="mode", mode="tools", target_profile=None))
        out = capsys.readouterr().out
        assert "could not be read as JSON" in out
        assert "Nothing was written" in out
        assert cfg_path.read_text(encoding="utf-8") == "{not json"

    def test_setup_refuses_before_asking_anything(self, monkeypatch, tmp_path, capsys):
        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text("{not json", encoding="utf-8")
        honcho_cli = self._point_at(monkeypatch, cfg_path)
        monkeypatch.setattr(honcho_cli, "_prompt", lambda *a, **k: pytest.fail("wizard asked a question"))
        honcho_cli.cmd_setup(SimpleNamespace())
        assert "Nothing was written" in capsys.readouterr().out
        assert cfg_path.read_text(encoding="utf-8") == "{not json"


class TestSetupApiKeyReplacesStaleGrant:
    """#97990: choosing apikey after a revoked grant left hosts.<name>.oauth in place, so the dead
    grant kept shadowing the fresh key across every later setup run."""

    def _cfg_with_grant(self):
        return {"hosts": {"hermes": {
            "apiKey": "hch-at-dead",
            "oauth": {"refreshToken": "hch-rt-dead", "expiresAt": 1, "clientId": "hermes-agent",
                      "tokenEndpoint": "https://api.honcho.dev/oauth/token"},
        }}}

    def _cloud_auth(self, monkeypatch, tmp_path, cfg, *, key_answer):
        import plugins.memory.honcho.cli as honcho_cli
        monkeypatch.setattr(honcho_cli, "_device_login_available", lambda: False)
        monkeypatch.setattr(honcho_cli, "_headless", lambda: (False, True))
        shown = []

        def _prompt(label, default=None, secret=False):
            shown.append(label)
            return "apikey" if "OAuth" in label else key_answer
        monkeypatch.setattr(honcho_cli, "_prompt", _prompt)
        ok = honcho_cli._setup_cloud_auth(cfg, cfg["hosts"]["hermes"], tmp_path / "honcho.json")
        return ok, cfg["hosts"]["hermes"]

    def test_new_key_clears_oauth_and_lands_on_the_host_block(self, monkeypatch, tmp_path):
        cfg = self._cfg_with_grant()
        ok, host = self._cloud_auth(monkeypatch, tmp_path, cfg, key_answer="hch-v3-fresh")
        assert ok is True
        assert "oauth" not in host
        assert host["apiKey"] == "hch-v3-fresh"
        assert cfg["apiKey"] == "hch-v3-fresh"

    def test_kept_root_key_still_clears_oauth(self, monkeypatch, tmp_path, capsys):
        cfg = self._cfg_with_grant()
        cfg["apiKey"] = "hch-v3-rootkey"
        ok, host = self._cloud_auth(monkeypatch, tmp_path, cfg, key_answer="")
        assert ok is True
        assert "oauth" not in host
        assert host["apiKey"] == "hch-v3-rootkey"
        assert "...-rootkey" in capsys.readouterr().out

    def test_dead_access_token_is_not_offered_as_current(self, monkeypatch, tmp_path, capsys):
        cfg = self._cfg_with_grant()
        ok, host = self._cloud_auth(monkeypatch, tmp_path, cfg, key_answer="")
        assert ok is False
        assert "Current API key: not set" in capsys.readouterr().out

    def test_static_host_key_without_grant_is_kept(self, monkeypatch, tmp_path):
        cfg = {"hosts": {"hermes": {"apiKey": "hch-v3-hostkey"}}}
        ok, host = self._cloud_auth(monkeypatch, tmp_path, cfg, key_answer="")
        assert ok is True
        assert host["apiKey"] == "hch-v3-hostkey"
        assert "apiKey" not in cfg


class TestEnabledRequiresACredential:
    """A host block must not be written with enabled: true unless it can authenticate. The default
    host's apiKey is not inherited by named profiles (#66125), so a clone of an OAuth-authenticated
    default block, or an enable on an empty block, has nothing to sign requests with."""

    def _env(self, monkeypatch, tmp_path, cfg, *, host="hermes_dreamer", profile="dreamer"):
        import plugins.memory.honcho.cli as honcho_cli
        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text("{}")
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        monkeypatch.setattr(honcho_cli, "_read_config", lambda: cfg)
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: host)
        monkeypatch.setattr(honcho_cli, "_active_profile_name", lambda: profile)
        monkeypatch.setattr(honcho_cli, "_ensure_peer_exists", lambda host_key=None: True)
        written = {}
        monkeypatch.setattr(honcho_cli, "_write_config", lambda c, path=None: written.setdefault("cfg", c))
        return honcho_cli, written

    def _oauth_default(self):
        return {"peerName": "eri", "hosts": {"hermes": {
            "enabled": True, "apiKey": "hch-at-live", "workspace": "hermes", "peerName": "eri",
            "oauth": {"refreshToken": "hch-rt-live", "expiresAt": 9e9, "clientId": "hermes-agent",
                      "tokenEndpoint": "https://api.honcho.dev/oauth/token"},
        }}}

    def test_clone_from_oauth_default_is_written_without_enabled(self, monkeypatch, tmp_path):
        honcho_cli, written = self._env(monkeypatch, tmp_path, self._oauth_default())
        assert honcho_cli.clone_honcho_for_profile("dreamer") is True
        block = written["cfg"]["hosts"]["hermes_dreamer"]
        assert "enabled" not in block
        assert "apiKey" not in block and "oauth" not in block
        assert block["aiPeer"] == "dreamer" and block["workspace"] == "hermes"

    def test_clone_from_host_only_static_key_is_written_without_enabled(self, monkeypatch, tmp_path):
        cfg = {"hosts": {"hermes": {"enabled": True, "apiKey": "hch-v3-hostonly", "workspace": "hermes"}}}
        honcho_cli, written = self._env(monkeypatch, tmp_path, cfg)
        assert honcho_cli.clone_honcho_for_profile("dreamer") is True
        assert "enabled" not in written["cfg"]["hosts"]["hermes_dreamer"]

    def test_clone_with_root_key_is_enabled(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "hch-v3-root", "hosts": {"hermes": {"workspace": "hermes"}}}
        honcho_cli, written = self._env(monkeypatch, tmp_path, cfg)
        assert honcho_cli.clone_honcho_for_profile("dreamer") is True
        assert written["cfg"]["hosts"]["hermes_dreamer"]["enabled"] is True

    def test_env_key_alone_does_not_enable_a_clone(self, monkeypatch, tmp_path):
        """A variable can vanish from the next process; only what is on disk counts at write time."""
        cfg = {"hosts": {"hermes": {"workspace": "hermes"}}}
        honcho_cli, written = self._env(monkeypatch, tmp_path, cfg)
        monkeypatch.setenv("HONCHO_API_KEY", "hch-v3-from-env")
        assert honcho_cli.clone_honcho_for_profile("dreamer") is True
        assert "enabled" not in written["cfg"]["hosts"]["hermes_dreamer"]

    def test_enable_refuses_with_only_an_env_key(self, monkeypatch, tmp_path, capsys):
        from types import SimpleNamespace
        cfg = {"hosts": {"hermes_dreamer": {"workspace": "hermes"}}}
        honcho_cli, written = self._env(monkeypatch, tmp_path, cfg)
        monkeypatch.setenv("HONCHO_API_KEY", "hch-v3-from-env")
        honcho_cli.cmd_enable(SimpleNamespace())
        assert written == {}
        assert "setup" in capsys.readouterr().out


    def test_clone_with_self_hosted_url_is_enabled(self, monkeypatch, tmp_path):
        cfg = {"baseUrl": "http://localhost:8000", "hosts": {"hermes": {"workspace": "hermes"}}}
        honcho_cli, written = self._env(monkeypatch, tmp_path, cfg)
        assert honcho_cli.clone_honcho_for_profile("dreamer") is True
        assert written["cfg"]["hosts"]["hermes_dreamer"]["enabled"] is True

    def test_enable_on_empty_block_refuses_and_writes_nothing(self, monkeypatch, tmp_path, capsys):
        honcho_cli, written = self._env(monkeypatch, tmp_path, self._oauth_default())
        honcho_cli.cmd_enable(SimpleNamespace())
        out = capsys.readouterr().out
        assert "stays disabled" in out
        assert "hermes honcho setup --target-profile dreamer" in out
        assert "hosts.hermes_dreamer" in out
        assert written == {}

    def test_enable_on_legacy_enabled_keyless_block_explains_instead_of_already_enabled(self, monkeypatch, tmp_path, capsys):
        cfg = self._oauth_default()
        cfg["hosts"]["hermes_dreamer"] = {"enabled": True, "aiPeer": "dreamer", "workspace": "hermes", "peerName": "eri"}
        honcho_cli, written = self._env(monkeypatch, tmp_path, cfg)
        honcho_cli.cmd_enable(SimpleNamespace())
        out = capsys.readouterr().out
        assert "stays disabled" in out
        assert "already enabled" not in out
        assert written == {}

    def test_enable_with_root_key_still_enables(self, monkeypatch, tmp_path, capsys):
        cfg = {"apiKey": "hch-v3-root", "hosts": {"hermes": {"workspace": "hermes"}}}
        honcho_cli, written = self._env(monkeypatch, tmp_path, cfg)
        honcho_cli.cmd_enable(SimpleNamespace())
        assert "Honcho enabled" in capsys.readouterr().out
        assert written["cfg"]["hosts"]["hermes_dreamer"]["enabled"] is True

    def test_enable_on_default_profile_hint_omits_target_profile(self, monkeypatch, tmp_path, capsys):
        honcho_cli, written = self._env(monkeypatch, tmp_path, {"hosts": {}}, host="hermes", profile="default")
        honcho_cli.cmd_enable(SimpleNamespace())
        out = capsys.readouterr().out
        assert "Run 'hermes honcho setup' to sign in" in out
        assert written == {}


class TestWriteConfigMergesOntoDisk:
    """A command reads honcho.json, works, then writes. A refresh in another process may have rotated
    the token in between; the write must keep that rotation and apply only the command's own edits."""

    def _paths(self, monkeypatch, tmp_path, disk):
        import plugins.memory.honcho.cli as honcho_cli
        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text(json.dumps(disk))
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        return honcho_cli, cfg_path

    def _rotated(self, cfg_path, disk):
        disk["hosts"]["hermes"]["apiKey"] = "hch-at-new"
        disk["hosts"]["hermes"]["oauth"] = {"refreshToken": "hch-rt-new"}
        cfg_path.write_text(json.dumps(disk))

    def test_rotation_between_read_and_write_survives(self, monkeypatch, tmp_path):
        disk = {"apiKey": "root", "hosts": {"hermes": {"apiKey": "hch-at-old", "oauth": {"refreshToken": "hch-rt-old"},
                                                     "recallMode": "hybrid"}}}
        honcho_cli, cfg_path = self._paths(monkeypatch, tmp_path, disk)
        cfg = honcho_cli._read_config()
        self._rotated(cfg_path, json.loads(cfg_path.read_text()))
        cfg["hosts"]["hermes"]["recallMode"] = "tools"
        cfg["dialecticCadence"] = 3
        honcho_cli._write_config(cfg)
        out = json.loads(cfg_path.read_text())
        assert out["hosts"]["hermes"]["apiKey"] == "hch-at-new"
        assert out["hosts"]["hermes"]["oauth"] == {"refreshToken": "hch-rt-new"}
        assert out["hosts"]["hermes"]["recallMode"] == "tools" and out["dialecticCadence"] == 3

    def test_a_credential_the_command_set_wins(self, monkeypatch, tmp_path):
        disk = {"hosts": {"hermes": {"apiKey": "hch-at-old", "oauth": {"refreshToken": "hch-rt-old"}}}}
        honcho_cli, cfg_path = self._paths(monkeypatch, tmp_path, disk)
        cfg = honcho_cli._read_config()
        self._rotated(cfg_path, json.loads(cfg_path.read_text()))
        cfg["hosts"]["hermes"]["apiKey"] = "hch-v3-pasted"
        cfg["hosts"]["hermes"].pop("oauth")
        honcho_cli._write_config(cfg)
        out = json.loads(cfg_path.read_text())
        assert out["hosts"]["hermes"]["apiKey"] == "hch-v3-pasted" and "oauth" not in out["hosts"]["hermes"]

    def test_a_key_the_command_removed_is_removed(self, monkeypatch, tmp_path):
        disk = {"hosts": {"hermes": {"apiKey": "k", "runtimePeerPrefix": "tg_"}}}
        honcho_cli, cfg_path = self._paths(monkeypatch, tmp_path, disk)
        cfg = honcho_cli._read_config()
        cfg["hosts"]["hermes"].pop("runtimePeerPrefix")
        honcho_cli._write_config(cfg)
        assert "runtimePeerPrefix" not in json.loads(cfg_path.read_text())["hosts"]["hermes"]

    def test_write_without_a_prior_read_writes_the_whole_config(self, monkeypatch, tmp_path):
        honcho_cli, cfg_path = self._paths(monkeypatch, tmp_path, {"hosts": {"hermes": {"apiKey": "k"}}})
        honcho_cli._write_config({"hosts": {"other": {"apiKey": "o"}}})
        assert json.loads(cfg_path.read_text()) == {"hosts": {"other": {"apiKey": "o"}}}

    def test_a_rebuilt_plain_dict_is_written_whole(self, monkeypatch, tmp_path):
        disk = {"hosts": {"hermes": {"apiKey": "hch-at-old", "recallMode": "hybrid"}}}
        honcho_cli, cfg_path = self._paths(monkeypatch, tmp_path, disk)
        cfg = dict(honcho_cli._read_config())
        self._rotated(cfg_path, json.loads(cfg_path.read_text()))
        honcho_cli._write_config(cfg)
        assert json.loads(cfg_path.read_text())["hosts"]["hermes"]["apiKey"] == "hch-at-old"

    def test_write_holds_the_refresh_file_lock(self, monkeypatch, tmp_path):
        import contextlib
        import plugins.memory.honcho.oauth as oauth
        honcho_cli, cfg_path = self._paths(monkeypatch, tmp_path, {})
        locked = []

        @contextlib.contextmanager
        def _lock(path):
            locked.append(path)
            yield

        monkeypatch.setattr(oauth, "_config_refresh_lock", _lock)
        honcho_cli._write_config({"apiKey": "k"})
        assert locked == [cfg_path]
