"""Tests for idempotent client onboarding (hermes_cli.add_client)."""

from pathlib import Path

import pytest
import yaml

from hermes_cli.add_client import (
    add_client,
    add_entry_to_doc,
    build_entry,
    derive_token_ref,
    load_doc,
    write_doc,
)
from hermes_cli.clients import RegistryError, load_registry


class TestBuildEntry:
    def test_derive_token_ref(self):
        assert derive_token_ref("acme") == "ACME_TG_TOKEN"
        assert derive_token_ref("petro-construction") == "PETRO_CONSTRUCTION_TG_TOKEN"

    def test_defaults_omitted(self):
        e = build_entry("acme", "prod")
        assert e == {"name": "acme", "env": "prod", "telegram_token_ref": "ACME_TG_TOKEN"}

    def test_non_default_fields_kept(self):
        e = build_entry("acme", "prod", model="m", tier="premium", isolation="container")
        assert e["model"] == "m" and e["tier"] == "premium" and e["isolation"] == "container"

    def test_default_profile_has_no_token(self):
        e = build_entry("default", "dev")
        assert "telegram_token_ref" not in e

    def test_explicit_profile_recorded(self):
        e = build_entry("acme", "prod", profile="acme-bot")
        assert e["profile"] == "acme-bot"


class TestDocOps:
    def test_add_then_duplicate_is_noop(self):
        doc = {"clients": []}
        assert add_entry_to_doc(doc, build_entry("acme", "prod")) is True
        assert add_entry_to_doc(doc, build_entry("acme", "dev")) is False
        assert len(doc["clients"]) == 1

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_doc(tmp_path / "nope.yaml") == {"clients": []}

    def test_write_rejects_invalid(self, tmp_path):
        # Two clients sharing a profile is invalid → write must refuse.
        doc = {"clients": [
            {"name": "a", "profile": "x", "env": "prod", "telegram_token_ref": "A"},
            {"name": "b", "profile": "x", "env": "prod", "telegram_token_ref": "B"},
        ]}
        with pytest.raises(RegistryError):
            write_doc(tmp_path / "clients.yaml", doc)
        assert not (tmp_path / "clients.yaml").exists()  # nothing persisted


class TestAddClientEndToEnd:
    def test_creates_registry_profile_and_secret(self, tmp_path):
        reg = tmp_path / "clients.yaml"
        root = tmp_path / "data"
        changed = add_client("acme", "prod", registry_path=reg, hermes_root=root)
        assert changed is True

        # registry now parses and contains acme
        loaded = load_registry(reg)
        assert "acme" in loaded.names
        # profile dir created
        assert (root / "profiles" / "acme").is_dir()
        # secret stub created, 0600, names the env var, no value
        secret = root / "secrets" / "acme.env"
        assert secret.is_file()
        assert oct(secret.stat().st_mode)[-3:] == "600"
        assert "ACME_TG_TOKEN=" in secret.read_text()

    def test_idempotent_rerun(self, tmp_path):
        reg = tmp_path / "clients.yaml"
        root = tmp_path / "data"
        add_client("acme", "prod", registry_path=reg, hermes_root=root)
        # write a real token, then re-run — must NOT be clobbered
        secret = root / "secrets" / "acme.env"
        secret.write_text("ACME_TG_TOKEN=12345:realtoken\n", encoding="utf-8")

        changed = add_client("acme", "prod", registry_path=reg, hermes_root=root)
        assert changed is False
        assert "realtoken" in secret.read_text()      # secret preserved
        assert len(load_registry(reg).clients) == 1    # no duplicate entry

    def test_rerun_fills_missing_profile_dir(self, tmp_path):
        reg = tmp_path / "clients.yaml"
        root = tmp_path / "data"
        add_client("acme", "prod", registry_path=reg, hermes_root=root)
        # simulate the profile dir going missing
        import shutil
        shutil.rmtree(root / "profiles" / "acme")

        add_client("acme", "prod", registry_path=reg, hermes_root=root)
        assert (root / "profiles" / "acme").is_dir()   # recreated

    def test_appends_to_existing_registry(self, tmp_path):
        reg = tmp_path / "clients.yaml"
        reg.write_text(yaml.safe_dump({"clients": [
            {"name": "default", "env": "dev"},
        ]}), encoding="utf-8")
        add_client("acme", "prod", registry_path=reg, hermes_root=tmp_path / "data")
        names = load_registry(reg).names
        assert names == ("default", "acme")

    def test_invalid_name_rejected(self, tmp_path):
        with pytest.raises(RegistryError):
            add_client("Bad_Name", "prod",
                       registry_path=tmp_path / "clients.yaml",
                       hermes_root=tmp_path / "data")

    def test_default_client_no_secret(self, tmp_path):
        root = tmp_path / "data"
        add_client("default", "dev",
                   registry_path=tmp_path / "clients.yaml", hermes_root=root)
        assert not (root / "secrets").exists()  # no token, no secret stub
