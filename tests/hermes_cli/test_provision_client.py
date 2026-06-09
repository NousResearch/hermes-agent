"""Tests for one-command client provisioning (hermes_cli.provision_client).

The live ``hermes profile create`` / ``hermes gateway restart`` steps are
exercised via an injected runner (they touch s6 / the container and can't run
in the sandbox); everything else — registry, the profile-.env token write, the
guard, idempotency, command shape — is verified directly.
"""

from pathlib import Path

import pytest

from hermes_cli.clients import load_registry
from hermes_cli.provision_client import (
    PROXY_KEY_VAR,
    TELEGRAM_TOKEN_VAR,
    build_create_command,
    build_restart_command,
    ensure_proxy_key,
    load_proxy_keys,
    profile_is_created,
    provision_client,
    token_value,
    write_token,
)


class TestTokenHelpers:
    def test_write_then_read(self, tmp_path):
        s = tmp_path / ".env"
        write_token(s, "12345:abc")
        assert token_value(s) == "12345:abc"
        assert oct(s.stat().st_mode)[-3:] == "600"

    def test_empty_value_is_none(self, tmp_path):
        s = tmp_path / ".env"
        s.write_text(f"{TELEGRAM_TOKEN_VAR}=\n", encoding="utf-8")
        assert token_value(s) is None

    def test_write_overrides_cloned_value_preserving_others(self, tmp_path):
        # Simulates a template-cloned .env carrying the template's token.
        s = tmp_path / ".env"
        s.write_text(f"OTHER=1\n{TELEGRAM_TOKEN_VAR}=template-token\n", encoding="utf-8")
        write_token(s, "client-token")
        body = s.read_text()
        assert "OTHER=1" in body
        assert token_value(s) == "client-token"
        assert "template-token" not in body

    def test_missing_file(self, tmp_path):
        assert token_value(tmp_path / "nope.env") is None


class TestCommandShape:
    def test_create_minimal(self):
        assert build_create_command("acme", clone_from=None, description=None) == [
            "hermes", "profile", "create", "acme",
        ]

    def test_create_clone_and_description(self):
        cmd = build_create_command("acme", clone_from="tmpl", description="sales bot")
        assert cmd == [
            "hermes", "profile", "create", "acme",
            "--clone", "--clone-from", "tmpl",
            "--description", "sales bot",
        ]

    def test_restart(self):
        assert build_restart_command("acme") == [
            "hermes", "gateway", "restart", "--profile", "acme",
        ]


class TestProvision:
    def _run(self, tmp_path, **kw):
        calls = []
        created = provision_client(
            "acme", "prod",
            registry_path=tmp_path / "clients.yaml",
            hermes_root=tmp_path / "data",
            runner=lambda argv: calls.append(list(argv)),
            **kw,
        )
        return created, calls

    def _profile_env(self, tmp_path):
        return tmp_path / "data" / "profiles" / "acme" / ".env"

    def test_token_guard_blocks_activation(self, tmp_path):
        with pytest.raises(ValueError, match="no Telegram token"):
            self._run(tmp_path)
        # registry entry is still recorded even though activation is refused
        assert "acme" in load_registry(tmp_path / "clients.yaml").names

    def test_allow_empty_token_stages_without_restart(self, tmp_path):
        created, calls = self._run(tmp_path, require_token=False)
        assert created is True
        # profile created, but NO Telegram token written and NO restart
        assert calls == [["hermes", "profile", "create", "acme"]]
        # the profile .env exists carrying only the minted proxy key (not the
        # Telegram token, which wasn't supplied)
        assert token_value(self._profile_env(tmp_path)) is None
        assert token_value(self._profile_env(tmp_path), var=PROXY_KEY_VAR)

    def test_token_writes_to_profile_env_and_restarts(self, tmp_path):
        created, calls = self._run(tmp_path, token="999:tok")
        assert created is True
        # token lands in the PROFILE's .env as TELEGRAM_BOT_TOKEN (what the
        # gateway actually reads) — not a separate secrets/<name>.env stub.
        assert token_value(self._profile_env(tmp_path)) == "999:tok"
        assert calls == [
            ["hermes", "profile", "create", "acme"],
            ["hermes", "gateway", "restart", "--profile", "acme"],
        ]

    def test_clone_from_passed_through(self, tmp_path):
        _, calls = self._run(tmp_path, token="t", clone_from="cheap-template")
        assert calls[0] == [
            "hermes", "profile", "create", "acme",
            "--clone", "--clone-from", "cheap-template",
        ]

    def test_idempotent_reconciles_token_without_recreating(self, tmp_path):
        # Simulate an already-created profile.
        pdir = tmp_path / "data" / "profiles" / "acme"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "config.yaml").write_text("model: x\n", encoding="utf-8")

        created, calls = self._run(tmp_path, token="newtok")
        assert created is False                         # not re-created
        # create NOT invoked; token still reconciled + gateway restarted
        assert calls == [["hermes", "gateway", "restart", "--profile", "acme"]]
        assert token_value(pdir / ".env") == "newtok"

    def test_existing_profile_token_satisfies_guard(self, tmp_path):
        # Profile already created with a token in its .env, no --token passed.
        pdir = tmp_path / "data" / "profiles" / "acme"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "config.yaml").write_text("model: x\n", encoding="utf-8")
        write_token(pdir / ".env", "pretok")

        created, calls = self._run(tmp_path)            # require_token default
        assert created is False
        assert calls == []                              # nothing to do

    def test_model_recorded_in_registry(self, tmp_path):
        self._run(tmp_path, token="t", model="deepseek-v4-flash")
        client = load_registry(tmp_path / "clients.yaml").get("acme")
        assert client.model == "deepseek-v4-flash"


class TestSharedTokenGuard:
    """The petro-construction footgun: ``hermes profile create --clone`` copies
    the template's .env verbatim and launches the gateway, so cloning a
    token-bearing template without an explicit --token would start a duplicate
    Telegram poller (409 'token already in use'). The guard refuses that."""

    def _seed_template(self, tmp_path, name="tmpl", *, token="111:AAA"):
        pdir = tmp_path / "data" / "profiles" / name
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "config.yaml").write_text("model: cheap\n", encoding="utf-8")
        lines = ["FOO=bar"]
        if token is not None:
            lines.insert(0, f"{TELEGRAM_TOKEN_VAR}={token}")
        (pdir / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return pdir

    def _call(self, tmp_path, **kw):
        calls = []
        created = provision_client(
            "acme", "prod",
            registry_path=tmp_path / "clients.yaml",
            hermes_root=tmp_path / "data",
            runner=lambda argv: calls.append(list(argv)),
            **kw,
        )
        return created, calls

    def test_refuses_clone_of_token_bearing_template_without_token(self, tmp_path):
        self._seed_template(tmp_path, token="111:AAA")
        calls = []
        with pytest.raises(ValueError, match="without --token"):
            provision_client(
                "acme", "prod",
                registry_path=tmp_path / "clients.yaml",
                hermes_root=tmp_path / "data",
                clone_from="tmpl", require_token=False,
                runner=lambda argv: calls.append(list(argv)),
            )
        # Fail-fast: nothing launched, and the registry was never written.
        assert calls == []
        assert not (tmp_path / "clients.yaml").exists()

    def test_refuses_clone_from_default_root_token(self, tmp_path):
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / ".env").write_text(
            f"{TELEGRAM_TOKEN_VAR}=111:AAA\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="without --token"):
            self._call(tmp_path, clone_from="default", require_token=False)

    def test_explicit_token_overrides_token_bearing_template(self, tmp_path):
        # --token is allowed even when the template carries one, and wins.
        self._seed_template(tmp_path, token="111:AAA")
        _, calls = self._call(tmp_path, token="999:BBB", clone_from="tmpl")
        env = tmp_path / "data" / "profiles" / "acme" / ".env"
        assert token_value(env) == "999:BBB"
        assert ["hermes", "gateway", "restart", "--profile", "acme"] in calls

    def test_tokenless_template_clone_stages_cleanly(self, tmp_path):
        # A template with no token is safe to clone-stage without --token.
        self._seed_template(tmp_path, token=None)
        _, calls = self._call(tmp_path, clone_from="tmpl", require_token=False)
        assert [
            "hermes", "profile", "create", "acme", "--clone", "--clone-from", "tmpl",
        ] in calls
        assert ["hermes", "gateway", "restart", "--profile", "acme"] not in calls


class TestProfileIsCreated:
    def test_detects_config(self, tmp_path):
        assert profile_is_created(tmp_path) is False
        (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
        assert profile_is_created(tmp_path) is True


class TestProxyKey:
    def test_mint_and_record_in_keys_json(self, tmp_path):
        key, created = ensure_proxy_key("acme", tmp_path)
        assert created is True
        assert key.startswith("hk-")
        keys = load_proxy_keys(tmp_path)
        assert keys == {key: "acme"}
        # keys.json is 0600
        kp = tmp_path / "proxy" / "keys.json"
        assert oct(kp.stat().st_mode)[-3:] == "600"

    def test_idempotent_reuses_existing_key(self, tmp_path):
        key1, c1 = ensure_proxy_key("acme", tmp_path)
        key2, c2 = ensure_proxy_key("acme", tmp_path)
        assert (c1, c2) == (True, False)
        assert key1 == key2
        assert load_proxy_keys(tmp_path) == {key1: "acme"}

    def test_distinct_clients_get_distinct_keys(self, tmp_path):
        ka, _ = ensure_proxy_key("acme", tmp_path)
        kb, _ = ensure_proxy_key("globex", tmp_path)
        assert ka != kb
        assert load_proxy_keys(tmp_path) == {ka: "acme", kb: "globex"}

    def test_provision_writes_key_to_keys_json_and_profile_env(self, tmp_path):
        calls = []
        provision_client(
            "acme", "prod",
            token="999:tok",
            registry_path=tmp_path / "clients.yaml",
            hermes_root=tmp_path / "data",
            runner=lambda argv: calls.append(list(argv)),
        )
        keys = load_proxy_keys(tmp_path / "data")
        assert list(keys.values()) == ["acme"]
        key = next(iter(keys))
        env = tmp_path / "data" / "profiles" / "acme" / ".env"
        assert token_value(env, var=PROXY_KEY_VAR) == key
        # Telegram token still lands correctly alongside the proxy key
        assert token_value(env) == "999:tok"

    def test_provision_is_idempotent_on_key(self, tmp_path):
        kw = dict(
            registry_path=tmp_path / "clients.yaml",
            hermes_root=tmp_path / "data",
            runner=lambda argv: None,
            token="t",
        )
        provision_client("acme", "prod", **kw)
        keys_first = load_proxy_keys(tmp_path / "data")
        provision_client("acme", "prod", **kw)
        keys_second = load_proxy_keys(tmp_path / "data")
        assert keys_first == keys_second               # no new key on re-run
