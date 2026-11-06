"""Autocomplete descriptions follow language-pack registration, not an import-time snapshot.

``hermes_cli.commands_completion`` used to bind ``COMMANDS`` with a module-level
``from hermes_cli.commands import COMMANDS``, freezing the English table built before
language-pack discovery finished; a pack registered afterwards never reached the menu
(#136743). The registry resolves ``COMMANDS`` per attribute access (PEP 562), so the
completer must keep going through the module instead of holding the first dict.
"""

from agent import i18n
from hermes_cli.commands_completion import SlashCommandCompleter
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document


def _menu_metas(text: str) -> dict[str, str]:
    """``{display: meta}`` for the given input — a fresh completer per call, as per keystroke."""
    return {
        item.display_text: item.display_meta_text
        for item in SlashCommandCompleter().get_completions(
            Document(text=text), CompleteEvent(completion_requested=True)
        )
    }


def _swapped_locale(tmp_path, monkeypatch):
    """Swap the bundled catalog for one whose ``/new`` copy differs from the English source."""
    fake = tmp_path / "locales"
    fake.mkdir()
    (fake / "en.yaml").write_text(
        "slash:\n"
        "  new:\n"
        "    description: 'SWAPPED-NEW-DESC'\n"
        "  shared:\n"
        "    usage_suffix: '{description} (usage: /{name} {args_hint})'\n"
        "    alias_for: '{description} (alias for /{name})'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake)
    i18n.reset_language_cache()


def test_menu_updates_when_a_pack_registers_after_import(tmp_path, monkeypatch):
    baseline = _menu_metas("/ne")
    assert "Start a new session" in baseline["/new"]

    _swapped_locale(tmp_path, monkeypatch)
    try:
        after = _menu_metas("/ne")
        assert "SWAPPED-NEW-DESC" in after["/new"]
        assert "Start a new session" not in after["/new"]
    finally:
        i18n.reset_language_cache()


def test_alias_entry_updates_too(tmp_path, monkeypatch):
    baseline = _menu_metas("/re")
    assert "Start a new session" in baseline["/reset"]  # "/reset" aliases "/new"

    _swapped_locale(tmp_path, monkeypatch)
    try:
        after = _menu_metas("/re")
        assert "SWAPPED-NEW-DESC" in after["/reset"]
        assert "alias for /new" in after["/reset"]
    finally:
        i18n.reset_language_cache()
