"""Tests for the declarative client (bot) registry — hermes_cli.clients.

Uses placeholder client names only (never real client identities), matching
the public-repo policy that real names live in a host-side clients.yaml.
"""

from pathlib import Path

import pytest

from hermes_cli.clients import (
    Client,
    Registry,
    RegistryError,
    load_registry,
    parse_registry,
    profile_dir,
    resolve_registry_path,
    secret_env_path,
)


def _valid_doc():
    return {
        "clients": [
            {"name": "default", "profile": "default", "env": "dev"},
            {"name": "acme", "env": "prod", "telegram_token_ref": "ACME_TG_TOKEN"},
            {
                "name": "beta-co",
                "env": "prod",
                "telegram_token_ref": "BETA_TG_TOKEN",
                "model": "claude-sonnet-4-6",
                "tier": "premium",
                "isolation": "container",
            },
        ]
    }


class TestParse:
    def test_valid_registry(self):
        reg = parse_registry(_valid_doc())
        assert reg.names == ("default", "acme", "beta-co")

    def test_profile_defaults_to_name(self):
        reg = parse_registry({"clients": [
            {"name": "acme", "env": "prod", "telegram_token_ref": "ACME_TG_TOKEN"},
        ]})
        assert reg.get("acme").profile == "acme"

    def test_optional_fields_default(self):
        c = parse_registry({"clients": [
            {"name": "acme", "env": "prod", "telegram_token_ref": "T"},
        ]}).get("acme")
        assert c.tier == "standard"
        assert c.isolation == "shared"
        assert c.model is None

    def test_default_profile_exempt_from_token(self):
        # 'default' may inherit the base config — no token_ref required.
        reg = parse_registry({"clients": [{"name": "default", "env": "dev"}]})
        assert reg.get("default").telegram_token_ref is None


class TestQueries:
    def test_for_env_and_profiles_for_env(self):
        reg = parse_registry(_valid_doc())
        assert reg.profiles_for_env("prod") == ("acme", "beta-co")
        assert reg.profiles_for_env("dev") == ("default",)
        assert [c.name for c in reg.for_env("prod")] == ["acme", "beta-co"]

    def test_get_missing_returns_none(self):
        assert parse_registry(_valid_doc()).get("nope") is None


class TestValidation:
    def test_missing_clients_key(self):
        with pytest.raises(RegistryError, match="missing the top-level 'clients'"):
            parse_registry({})

    def test_clients_not_a_list(self):
        with pytest.raises(RegistryError, match="'clients' must be a list"):
            parse_registry({"clients": {"name": "acme"}})

    def test_duplicate_name(self):
        doc = {"clients": [
            {"name": "acme", "env": "prod", "telegram_token_ref": "T"},
            {"name": "acme", "env": "dev", "telegram_token_ref": "T2"},
        ]}
        with pytest.raises(RegistryError, match="duplicate client name 'acme'"):
            parse_registry(doc)

    def test_duplicate_profile(self):
        doc = {"clients": [
            {"name": "acme", "profile": "shared", "env": "prod", "telegram_token_ref": "T"},
            {"name": "beta", "profile": "shared", "env": "prod", "telegram_token_ref": "T2"},
        ]}
        with pytest.raises(RegistryError, match="already used by another client"):
            parse_registry(doc)

    def test_invalid_env(self):
        with pytest.raises(RegistryError, match="invalid env"):
            parse_registry({"clients": [
                {"name": "acme", "env": "staging", "telegram_token_ref": "T"},
            ]})

    def test_invalid_isolation(self):
        with pytest.raises(RegistryError, match="invalid isolation"):
            parse_registry({"clients": [
                {"name": "acme", "env": "prod", "telegram_token_ref": "T", "isolation": "vm"},
            ]})

    def test_invalid_name(self):
        with pytest.raises(RegistryError, match="is invalid"):
            parse_registry({"clients": [
                {"name": "Acme_Corp", "env": "prod", "telegram_token_ref": "T"},
            ]})

    def test_named_bot_requires_token_ref(self):
        with pytest.raises(RegistryError, match="missing 'telegram_token_ref'"):
            parse_registry({"clients": [{"name": "acme", "env": "prod"}]})

    def test_errors_aggregated(self):
        # Two distinct problems should both appear in one raise.
        doc = {"clients": [
            {"name": "acme", "env": "staging", "telegram_token_ref": "T"},
            {"name": "BadName", "env": "prod", "telegram_token_ref": "T2"},
        ]}
        with pytest.raises(RegistryError) as exc:
            parse_registry(doc)
        msg = str(exc.value)
        assert "invalid env" in msg and "is invalid" in msg


class TestPaths:
    def test_profile_dir_named_vs_default(self):
        reg = parse_registry(_valid_doc())
        root = Path("/opt/data")
        assert profile_dir(reg.get("default"), root) == root
        assert profile_dir(reg.get("acme"), root) == root / "profiles" / "acme"

    def test_secret_env_path(self):
        c = parse_registry(_valid_doc()).get("acme")
        assert secret_env_path(c, "/opt/data") == Path("/opt/data/secrets/acme.env")


class TestLoadAndResolve:
    def test_resolve_explicit_path(self):
        assert resolve_registry_path("/tmp/x.yaml") == Path("/tmp/x.yaml")

    def test_resolve_env_var(self, monkeypatch):
        monkeypatch.setenv("HERMES_CLIENTS_REGISTRY", "/etc/clients.yaml")
        assert resolve_registry_path() == Path("/etc/clients.yaml")

    def test_resolve_none_raises(self, monkeypatch):
        monkeypatch.delenv("HERMES_CLIENTS_REGISTRY", raising=False)
        with pytest.raises(RegistryError, match="no registry path"):
            resolve_registry_path()

    def test_load_roundtrip(self, tmp_path):
        import yaml

        f = tmp_path / "clients.yaml"
        f.write_text(yaml.safe_dump(_valid_doc()), encoding="utf-8")
        reg = load_registry(f)
        assert reg.names == ("default", "acme", "beta-co")

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(RegistryError, match="not found"):
            load_registry(tmp_path / "nope.yaml")

    def test_shipped_example_is_valid(self):
        # clients.example.yaml in the repo root must always parse cleanly.
        example = Path(__file__).resolve().parents[2] / "clients.example.yaml"
        reg = load_registry(example)
        assert "acme" in reg.names
