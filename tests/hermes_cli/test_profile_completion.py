"""/profile completes profile names — one provider in the shared completer, so the CLI,
the TUI and the desktop (``complete.slash``) all get the list from the same place — and
the argument a completed name lands in does something: ``--list`` prints the roster, a name
prints that profile's details plus the command that actually switches to it.

The provider reads real profile dirs (a config parse per profile), so these tests stub
``hermes_cli.profiles`` rather than depend on whatever profiles this box happens to have.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hermes_cli.commands_completion as completion_mod
from hermes_cli.commands import SUBCOMMANDS, command_desktop_meta, resolve_command
from hermes_cli.slash_exec import CommandContext, execute_command
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text


def _profile(name: str, model: str | None, is_default: bool = False, *,
             provider: str | None = None, path: str | None = None):
    return SimpleNamespace(name=name, model=model, provider=provider, is_default=is_default,
                           path=Path(path or f"/profiles/{name}"), display_name="")


def _names(text: str) -> list[tuple[str, str]]:
    completer = completion_mod.SlashCommandCompleter()
    return [(item.text, to_plain_text(item.display_meta) if item.display_meta else "") for item in
            completer.get_completions(Document(text, len(text)), None)]


def test_profile_lists_names_with_model_meta(monkeypatch):
    monkeypatch.setattr(completion_mod, "_profiles_memo", None)
    monkeypatch.setattr("hermes_cli.profiles.list_profiles",
                        lambda **_kw: [_profile("default", "cheap-ai-fallback", True),
                                       _profile("coder", "qwen3.8-flash")])

    rows = _names("/profile ")

    assert [name for name, _meta in rows] == ["default", "coder"]
    assert rows[0][1] == "cheap-ai-fallback · default"
    assert rows[1][1] == "qwen3.8-flash"


def test_profile_filters_by_typed_prefix(monkeypatch):
    monkeypatch.setattr(completion_mod, "_profiles_memo", None)
    monkeypatch.setattr("hermes_cli.profiles.list_profiles",
                        lambda **_kw: [_profile("default", None, True), _profile("coder", None)])

    assert [name for name, _meta in _names("/profile co")] == ["coder"]
    assert _names("/profile zz") == []


def test_profile_listing_is_memoised_until_the_root_changes(monkeypatch):
    """Per-keystroke completion must not re-parse every profile's config.yaml."""
    calls = []

    def listing(**_kw):
        calls.append(1)
        return [_profile("default", None, True)]

    monkeypatch.setattr(completion_mod, "_profiles_memo", None)
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", listing)
    monkeypatch.setattr(completion_mod, "_profiles_signature", lambda: ("root", 1, 1))

    _names("/profile ")
    _names("/profile ")
    assert len(calls) == 1

    # A create/delete/rename touches the profiles root: the next keystroke must see it.
    monkeypatch.setattr(completion_mod, "_profiles_signature", lambda: ("root", 2, 1))
    _names("/profile ")
    assert len(calls) == 2


def test_profile_falls_back_to_names_when_a_config_will_not_load(monkeypatch):
    monkeypatch.setattr(completion_mod, "_profiles_memo", None)

    def boom(**_kw):
        raise ValueError("bad config.yaml")

    monkeypatch.setattr("hermes_cli.profiles.list_profiles", boom)
    monkeypatch.setattr("hermes_cli.profiles.list_profile_names", lambda: ["default", "coder"])

    assert [name for name, _meta in _names("/profile ")] == ["default", "coder"]


def test_profile_registry_keeps_the_arg_stage_open():
    """The desktop only stays in the arg stage for an options/mixed mode; without this
    ``/profile `` closes the popover and the names are unreachable there."""
    profile = resolve_command("profile")

    assert profile is not None
    assert command_desktop_meta(profile)["argument_mode"] == "options"
    assert profile.args_hint == "[name | --list]"
    # `--list` must not register as a pipe-group subcommand: SUBCOMMANDS derives from
    # args_hint via `[a-z]+(\|[a-z]+)+`, which would ghost-text "name"/"--list" words.
    assert "profile" not in SUBCOMMANDS


# ---------------------------------------------------------------------------
# Execution: /profile <arg>. The argument used to be ignored, so a name the popover
# offered was a dead end. Bare /profile keeps its old text; `--list` and a name now read.
# ---------------------------------------------------------------------------

def _run(arg: str, *, active: str = "default", surface: str = "cli"):
    return execute_command("profile", CommandContext(
        surface=surface, args=arg, options={"profile_name": active, "home_display": "~/.hermes"}))


def _stub_profiles(monkeypatch, *profiles):
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda **_kw: list(profiles))


def test_bare_profile_still_reports_the_active_profile():
    reply = _run("")

    assert reply.data == {"profile": "default", "home": "~/.hermes"}


def test_profile_list_prints_the_roster_with_the_active_one_marked(monkeypatch):
    _stub_profiles(monkeypatch, _profile("default", "cheap-ai-fallback", True),
                   _profile("coder", "qwen3.8-flash"))

    reply = _run("--list", active="coder")

    assert reply.data["profiles"] == ["default", "coder"]
    assert reply.data["active"] == "coder"
    lines = reply.text.splitlines()
    assert any("coder" in line and "◆" in line for line in lines)
    assert not any("default" in line and "◆" in line for line in lines)
    assert "hermes -p <name>" in reply.text  # the roster says how a profile is actually entered


def test_profile_list_accepts_the_short_flag(monkeypatch):
    _stub_profiles(monkeypatch, _profile("default", None, True))

    assert _run("-l").text == _run("--list").text


def test_a_profile_named_list_is_still_a_profile(monkeypatch):
    """``PROFILE_ID_RE`` allows ``list``, so the roster is a flag — the word stays a name."""
    _stub_profiles(monkeypatch, _profile("default", None, True),
                   _profile("list", "qwen3.8-flash", provider="openrouter"))

    reply = _run("list")

    assert reply.data["profile"] == "list"
    assert "Model:   qwen3.8-flash (openrouter)" in reply.text
    assert "Profiles (" not in reply.text


def test_profile_name_prints_details_and_the_command_that_switches(monkeypatch):
    _stub_profiles(monkeypatch, _profile("default", "cheap-ai-fallback", True),
                   _profile("coder", "qwen3.8-flash", provider="openrouter", path="/p/coder"))

    reply = _run("coder", active="default")

    assert "Model:   qwen3.8-flash (openrouter)" in reply.text
    assert "Path:    /p/coder" in reply.text
    assert "Active:  no" in reply.text
    assert "hermes -p coder" in reply.text
    assert "hermes profile use coder" in reply.text
    assert reply.data["profile"] == "coder"
    assert reply.data["active"] is False


def test_profile_name_says_when_it_is_already_active(monkeypatch):
    _stub_profiles(monkeypatch, _profile("coder", "qwen3.8-flash", path="/p/coder"))

    reply = _run("coder", active="coder")

    assert "Active:  yes" in reply.text
    assert reply.data["active"] is True


def test_profile_unknown_name_lists_what_exists(monkeypatch):
    _stub_profiles(monkeypatch, _profile("default", None, True), _profile("coder", None))

    reply = _run("nope")

    assert "No profile named 'nope'" in reply.text
    assert "default, coder" in reply.text
    assert reply.data["error"] == "unknown_profile"
    assert reply.data["requested"] == "nope"


def test_profile_argument_text_is_surface_independent(monkeypatch):
    """The executor's contract: same args/options -> same text on every surface."""
    _stub_profiles(monkeypatch, _profile("default", None, True), _profile("coder", "m"))

    assert _run("list", surface="cli").text == _run("list", surface="gateway").text
    assert _run("coder", surface="cli").text == _run("coder", surface="tui").text


def test_classic_cli_forwards_the_argument_to_the_executor():
    """The dispatch table must hand /profile its argument; ``False`` silently dropped it."""
    from cli import HermesCLI

    assert HermesCLI._slash_handler("profile") == ("_handle_profile_command", True)
