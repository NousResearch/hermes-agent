"""``/creds`` handlers: subcommand routing, overwrite policy, env injection safety."""

import os

import pytest

import agent.credential_store as cs
from hermes_cli.credential_commands import handle_creds_command


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the store at a per-test home and drop any cached instance."""
    monkeypatch.setattr(cs, "get_hermes_home", lambda: tmp_path)
    cs.reset_store()
    yield
    cs.reset_store()


def test_bare_command_and_list_show_names_not_values():
    assert "No credentials stored" in handle_creds_command("")
    assert "Usage:" in handle_creds_command("set")

    assert "Stored 'supabase_service_key'" in handle_creds_command("set supabase_service_key sk-live-abc123")
    listing = handle_creds_command("list")
    assert "supabase_service_key" in listing
    assert "sk-live-abc123" not in listing  # names only

    assert handle_creds_command("show supabase_service_key") == "supabase_service_key: sk-live-abc123"
    assert "not found" in handle_creds_command("show missing_credential_name")


def test_set_requires_force_to_replace():
    handle_creds_command("set stripe_secret_key first_value")
    denied = handle_creds_command("set stripe_secret_key second_value")
    assert "already exists" in denied
    assert "--force" in denied
    assert cs.get_store().get("stripe_secret_key") == "first_value"

    assert "Updated" in handle_creds_command("set stripe_secret_key second_value --force")
    assert cs.get_store().get("stripe_secret_key") == "second_value"


def test_set_rejects_generic_names_and_empty_values():
    assert "too generic" in handle_creds_command("set key value")
    assert "at least 3" in handle_creds_command("set ab value")


def test_clear_needs_confirmation():
    handle_creds_command("set alpha_token_one value1")
    refused = handle_creds_command("clear")
    assert "--yes" in refused
    assert cs.get_store().list()  # untouched

    assert "Cleared 1 credential(s)" in handle_creds_command("clear --yes")
    assert cs.get_store().list() == []


def test_remove_and_alias():
    handle_creds_command("set alpha_token_one value1")
    assert "Removed 'alpha_token_one'" in handle_creds_command("remove alpha_token_one")
    handle_creds_command("set beta_token_two value2")
    assert "Removed 'beta_token_two'" in handle_creds_command("delete beta_token_two")


def test_inject_sets_env_and_unset_removes_it(monkeypatch):
    handle_creds_command("set supabase_service_key sk-live-abc123")
    out = handle_creds_command("inject supabase_service_key")
    assert "SUPABASE_SERVICE_KEY" in out
    assert os.environ.get("SUPABASE_SERVICE_KEY") == "sk-live-abc123"
    assert "Not found: nope_credential" in handle_creds_command("inject nope_credential")

    assert "Unset (1): SUPABASE_SERVICE_KEY" in handle_creds_command("unset supabase_service_key")
    assert "SUPABASE_SERVICE_KEY" not in os.environ


def test_inject_refuses_process_hijacking_names():
    # Generic names are rejected at the store layer; hijack-shaped env names are
    # refused at the inject layer even when they were stored under a valid name.
    assert "too generic" in handle_creds_command("set path abc12345")

    os.environ.pop("LD_PRELOAD", None)
    handle_creds_command("set ld_preload abc12345")
    out = handle_creds_command("inject ld_preload")
    assert "refused" in out.lower()
    assert "LD_PRELOAD" not in os.environ

    handle_creds_command("set run_secret_value abc12345")
    os.environ.pop("RUN_SECRET_VALUE", None)
    out = handle_creds_command("inject run_secret_value")
    assert "RUN_SECRET_VALUE" in out
    assert os.environ.pop("RUN_SECRET_VALUE") == "abc12345"


def test_unknown_subcommand_prints_usage():
    assert "Unknown subcommand: /creds frobnicate" in handle_creds_command("frobnicate")
    assert "/creds clear --yes" in handle_creds_command("frobnicate")


def test_registry_wiring_keeps_creds_off_the_gateway():
    """CommandDef + executor + gateway exclusion (no CLI import: see dispatch test)."""
    from hermes_cli.commands import gateway_help_lines, resolve_command
    from hermes_cli.slash_exec import resolve_executor

    cmd_def = resolve_command("creds")
    assert cmd_def is not None and cmd_def.name == "creds"
    assert resolve_command("credentials") is cmd_def          # alias
    assert cmd_def.cli_only is True                            # `show` must not reach chat
    assert cmd_def.execute == "creds"
    assert resolve_executor(cmd_def) is not None
    assert not any("creds" in line for line in gateway_help_lines())


def test_cli_dispatch_routes_creds_to_the_handler():
    """The REPL dispatcher must resolve /creds to the handler method by name."""
    from cli import HermesCLI

    assert HermesCLI._slash_handler("creds") == ("_handle_creds_command", True)


def test_corrupt_store_reports_failure_and_changes_nothing(tmp_path):
    handle_creds_command("set alpha_token_one value1")
    store = cs.get_store()
    store._store_path.write_bytes(b"\x00broken")
    out = handle_creds_command("set beta_token_two value2")
    assert "Nothing was changed" in out
    assert store._store_path.read_bytes() == b"\x00broken"