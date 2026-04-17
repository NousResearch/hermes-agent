import importlib
import sys
from unittest.mock import MagicMock, patch


def _load_cli_module():
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
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False
    ):
        import cli as cli_mod

        return importlib.reload(cli_mod)


def test_normalize_voice_record_key_keeps_valid_ctrl_binding():
    cli_mod = _load_cli_module()

    binding, display = cli_mod._normalize_voice_record_key(
        "ctrl+space",
        validator=lambda _binding: None,
    )

    assert binding == "c-space"
    assert display == "Ctrl+Space"


def test_get_voice_record_key_binding_falls_back_on_alt_modifier():
    cli_mod = _load_cli_module()

    binding, display = cli_mod._get_voice_record_key_binding(
        load_config_fn=lambda: {"voice": {"record_key": "alt+space"}},
        validator=lambda _binding: None,
    )

    assert binding == "c-b"
    assert display == "Ctrl+B"


def test_get_voice_record_key_binding_falls_back_when_validator_rejects_binding():
    cli_mod = _load_cli_module()

    def _reject(_binding: str) -> None:
        if _binding == "s-space":
            raise ValueError("Invalid key")

    binding, display = cli_mod._get_voice_record_key_binding(
        load_config_fn=lambda: {"voice": {"record_key": "shift+space"}},
        validator=_reject,
    )

    assert binding == "c-b"
    assert display == "Ctrl+B"
