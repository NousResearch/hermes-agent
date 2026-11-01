"""CLI info/help copy resolves through the i18n catalog: /help consumes ``CommandDef.describe()``,
module-level label tables became call-time lookups, and error copy follows the active language."""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml
import pytest

from agent import i18n
from agent.i18n import t
from hermes_cli.commands import COMMAND_REGISTRY, CommandDef
from hermes_cli import cli_info_mixin
from hermes_cli.cli_chat_error_copy import agent_init_failure_message, chat_error_response
from hermes_cli.cli_unknown_command import unknown_command_lines


@pytest.fixture(autouse=True)
def _fresh_language_cache():
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def test_describe_falls_back_to_english_field_when_no_catalog_entry():
    cmd = CommandDef("no-such-command-zz", "Does a thing", "Info")
    assert cmd.describe() == "Does a thing"


def test_describe_uses_catalog_entry_when_present(tmp_path, monkeypatch):
    (tmp_path / "en.yaml").write_text("slash:\n  help:\n    description: 'Localized help'\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: tmp_path)
    i18n.reset_language_cache()
    try:
        help_cmd = next(c for c in COMMAND_REGISTRY if c.name == "help")
        assert help_cmd.describe() == "Localized help"
    finally:
        i18n.reset_language_cache()


def test_registry_fallback_descriptions_match_bundled_english_catalog():
    """Every canonical registry fallback must match a real bundled English entry."""
    locales_dir = Path(__file__).resolve().parents[2] / "locales"
    with (locales_dir / "en.yaml").open("r", encoding="utf-8") as catalog_file:
        slash_catalog = (yaml.safe_load(catalog_file) or {}).get("slash", {})

    mismatches = {}
    for cmd in COMMAND_REGISTRY:
        entry = slash_catalog.get(cmd.name)
        catalog_description = entry.get("description") if isinstance(entry, dict) else None
        if catalog_description != cmd.description:
            mismatches[cmd.name] = {
                "registry": cmd.description,
                "catalog": catalog_description,
            }

    assert mismatches == {}


def test_tool_progress_label_is_localized_per_mode_and_empty_for_unknown():
    for mode in ("off", "new", "all", "verbose"):
        line = cli_info_mixin._tool_progress_label(mode)
        assert t(f"cli.verbose.label_{mode}") in line
        assert t(f"cli.verbose.detail_{mode}") in line
    assert cli_info_mixin._tool_progress_label("nope") == ""


def test_reload_mcp_choices_carry_stable_ids_and_catalog_labels():
    choices = cli_info_mixin._reload_mcp_choices()
    assert [c[0] for c in choices] == ["once", "always", "cancel"]
    assert choices[0][1] == t("cli.reload_mcp.choice_once")


def test_help_section_titles_localize_known_categories_and_pass_unknown_through():
    assert cli_info_mixin._help_section_title("Tools & Skills") == t("cli.help.section_tools_skills")
    assert cli_info_mixin._help_section_title("Custom Plugin Category") == "Custom Plugin Category"


def test_chat_error_copy_uses_catalog_lead_and_details():
    text = chat_error_response("upstream said no", provider="openrouter", model="m", failure_reason="rate_limit")
    assert text == t("cli.error.with_details",
                     lead=t("cli.error.rate_limit", provider="openrouter", model="m"),
                     details="upstream said no")
    assert agent_init_failure_message(RuntimeError("boom")) == t("cli.error.agent_init_failed", error="boom")


def test_unknown_command_lines_localize_lead_and_pointer():
    lead, pointer = unknown_command_lines("/modle", {"/model", "/help"})
    hint = " " + t("cli.command.did_you_mean_one", suggestion="/model")
    assert lead == t("cli.command.unknown_nothing_sent", command="/modle", hint=hint)
    assert pointer == t("cli.command.type_help")
