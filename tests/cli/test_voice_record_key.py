import importlib
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _import_cli_module():
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
        "fire": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as mod

        return importlib.reload(mod)


class TestVoiceRecordKeyConfig:
    def test_empty_record_key_sets_hands_free_vad_cache(self):
        mod = _import_cli_module()
        cli = mod.HermesCLI.__new__(mod.HermesCLI)

        cli.set_voice_record_key_cache("")

        assert cli._voice_record_key_display_cache == "hands-free (VAD)"
        assert cli._voice_record_key_vad_cache is True
        assert cli._voice_record_key_label() == "hands-free (VAD)"

    def test_non_empty_record_key_uses_normal_formatter(self):
        mod = _import_cli_module()
        cli = mod.HermesCLI.__new__(mod.HermesCLI)

        cli.set_voice_record_key_cache("ctrl+o")

        assert cli._voice_record_key_display_cache == "Ctrl+O"
        assert cli._voice_record_key_vad_cache is False
        assert cli._voice_record_key_label() == "Ctrl+O"

    def test_optional_keybinding_skips_registration_for_vad_mode(self):
        mod = _import_cli_module()
        kb = MagicMock()
        handler = MagicMock()

        mod._register_optional_keybinding(kb, None, handler)

        kb.add.assert_not_called()

    def test_optional_keybinding_registers_non_empty_key(self):
        mod = _import_cli_module()
        kb = MagicMock()
        decorator = MagicMock()
        kb.add.return_value = decorator
        handler = MagicMock()

        mod._register_optional_keybinding(kb, "c-b", handler)

        kb.add.assert_called_once_with("c-b")
        decorator.assert_called_once_with(handler)
