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
    TELEGRAM_TOKEN_VAR,
    build_create_command,
    build_restart_command,
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
        # profile created, but NO token written and NO restart
        assert calls == [["hermes", "profile", "create", "acme"]]
        assert not self._profile_env(tmp_path).exists()

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


class TestProfileIsCreated:
    def test_detects_config(self, tmp_path):
        assert profile_is_created(tmp_path) is False
        (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
        assert profile_is_created(tmp_path) is True
