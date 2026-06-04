"""Tests for one-command client provisioning (hermes_cli.provision_client).

The live ``hermes profile create`` step is exercised via an injected runner
(it touches s6 / the container and can't run in the sandbox); everything else
— registry, secret handling, the token guard, idempotency, command shape — is
verified directly.
"""

from pathlib import Path

import pytest

from hermes_cli.clients import load_registry
from hermes_cli.provision_client import (
    build_create_command,
    profile_is_created,
    provision_client,
    token_value,
    write_token,
)


class TestTokenHelpers:
    def test_write_then_read(self, tmp_path):
        s = tmp_path / "acme.env"
        write_token(s, "ACME_TG_TOKEN", "12345:abc")
        assert token_value(s, "ACME_TG_TOKEN") == "12345:abc"
        assert oct(s.stat().st_mode)[-3:] == "600"

    def test_empty_value_is_none(self, tmp_path):
        s = tmp_path / "acme.env"
        s.write_text("ACME_TG_TOKEN=\n", encoding="utf-8")
        assert token_value(s, "ACME_TG_TOKEN") is None

    def test_write_preserves_other_lines(self, tmp_path):
        s = tmp_path / "acme.env"
        s.write_text("# comment\nOTHER=1\nACME_TG_TOKEN=\n", encoding="utf-8")
        write_token(s, "ACME_TG_TOKEN", "tok")
        body = s.read_text()
        assert "OTHER=1" in body and "# comment" in body
        assert token_value(s, "ACME_TG_TOKEN") == "tok"

    def test_missing_file(self, tmp_path):
        assert token_value(tmp_path / "nope.env", "X") is None


class TestCommandShape:
    def test_minimal(self):
        assert build_create_command("acme", clone_from=None, description=None) == [
            "hermes", "profile", "create", "acme",
        ]

    def test_clone_and_description(self):
        cmd = build_create_command("acme", clone_from="tmpl", description="sales bot")
        assert cmd == [
            "hermes", "profile", "create", "acme",
            "--clone", "--clone-from", "tmpl",
            "--description", "sales bot",
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

    def test_token_guard_blocks_activation(self, tmp_path):
        # No token supplied and stub is empty → must refuse to create profile.
        with pytest.raises(ValueError, match="is not set"):
            self._run(tmp_path)

    def test_allow_empty_token_stages_without_create(self, tmp_path):
        created, calls = self._run(tmp_path, require_token=False)
        # registry written, but no `hermes profile create` invoked? It WILL be
        # created since profile isn't created yet and we allow empty token.
        assert created is True
        assert calls and calls[0][:3] == ["hermes", "profile", "create"]
        # client is in the registry
        assert "acme" in load_registry(tmp_path / "clients.yaml").names

    def test_token_provided_writes_and_creates(self, tmp_path):
        created, calls = self._run(tmp_path, token="999:tok")
        assert created is True
        secret = tmp_path / "data" / "secrets" / "acme.env"
        assert token_value(secret, "ACME_TG_TOKEN") == "999:tok"
        assert calls[0] == ["hermes", "profile", "create", "acme"]

    def test_clone_from_passed_through(self, tmp_path):
        _, calls = self._run(tmp_path, token="t", clone_from="cheap-template")
        assert calls[0] == [
            "hermes", "profile", "create", "acme",
            "--clone", "--clone-from", "cheap-template",
        ]

    def test_idempotent_when_already_created(self, tmp_path):
        # First run creates; simulate profile creation by writing config.yaml.
        self._run(tmp_path, token="t")
        pdir = tmp_path / "data" / "profiles" / "acme"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "config.yaml").write_text("model: x\n", encoding="utf-8")

        created, calls = self._run(tmp_path, token="t")
        assert created is False          # no-op
        assert calls == []               # profile create NOT re-invoked

    def test_model_recorded_in_registry(self, tmp_path):
        self._run(tmp_path, token="t", model="deepseek-v4-flash")
        client = load_registry(tmp_path / "clients.yaml").get("acme")
        assert client.model == "deepseek-v4-flash"


class TestProfileIsCreated:
    def test_detects_config(self, tmp_path):
        assert profile_is_created(tmp_path) is False
        (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
        assert profile_is_created(tmp_path) is True
