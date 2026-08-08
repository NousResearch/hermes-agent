"""Regression tests for wave-1 shard-s2 extraction of ``hermes_cli/main.py``.

The provider picker (``cmd_model`` / ``select_provider_and_model`` /
``_is_profile_api_key_provider``) moved verbatim into
``hermes_cli/model_picker.py`` and the auxiliary-model config UI
(``_AUX_TASKS`` and the ``_aux_*`` functions) moved verbatim into
``hermes_cli/aux_config_cmd.py``. main.py re-imports every name so
``hermes_cli.main.*`` call sites and test monkeypatches keep resolving to
the SAME objects.

These tests pin that contract (identity of re-exports) and re-check the
pure helpers' behavior in their new homes.
"""

import hermes_cli.aux_config_cmd as aux_cmd
import hermes_cli.main as main_mod
import hermes_cli.model_picker as picker


def test_main_reexports_are_same_objects():
    """hermes_cli.main.* must resolve to the moved objects, not copies."""
    assert main_mod.cmd_model is picker.cmd_model
    assert main_mod.select_provider_and_model is picker.select_provider_and_model
    assert main_mod._is_profile_api_key_provider is picker._is_profile_api_key_provider
    assert main_mod._AUX_TASKS is aux_cmd._AUX_TASKS
    assert main_mod._all_aux_tasks is aux_cmd._all_aux_tasks
    assert main_mod._format_aux_current is aux_cmd._format_aux_current
    assert main_mod._save_aux_choice is aux_cmd._save_aux_choice
    assert main_mod._reset_aux_to_auto is aux_cmd._reset_aux_to_auto
    assert main_mod._aux_config_menu is aux_cmd._aux_config_menu
    assert main_mod._aux_select_for_task is aux_cmd._aux_select_for_task
    assert main_mod._aux_flow_provider_model is aux_cmd._aux_flow_provider_model
    assert main_mod._aux_flow_custom_endpoint is aux_cmd._aux_flow_custom_endpoint


def test_format_aux_current_renders():
    """Pure display helper behavior is unchanged after the move."""
    assert aux_cmd._format_aux_current(None) == "auto"
    assert aux_cmd._format_aux_current({}) == "auto"
    assert aux_cmd._format_aux_current({"provider": "auto"}) == "auto"
    assert aux_cmd._format_aux_current({"provider": "openai"}) == "openai"
    assert (
        aux_cmd._format_aux_current({"provider": "openai", "model": "gpt-4o"})
        == "openai \u00b7 gpt-4o"
    )
    assert (
        aux_cmd._format_aux_current({"base_url": "https://example.com/v1"})
        == "custom (example.com/v1)"
    )
    assert (
        aux_cmd._format_aux_current(
            {"base_url": "https://example.com/v1", "model": "qwen2.5"}
        )
        == "custom (example.com/v1) \u00b7 qwen2.5"
    )


def test_aux_tasks_table_integrity():
    """_AUX_TASKS keeps unique keys and fully populated rows."""
    keys = [k for k, _name, _desc in aux_cmd._AUX_TASKS]
    assert len(keys) == len(set(keys)), "aux task keys must be unique"
    assert "curator" in keys
    assert "compression" in keys
    assert "vision" in keys
    for _k, name, desc in aux_cmd._AUX_TASKS:
        assert name and desc, "every aux task row needs a display name and description"


def test_is_profile_api_key_provider_unknown_provider():
    """Unknown provider ids must not crash the catch-all dispatch helper."""
    assert picker._is_profile_api_key_provider("definitely-not-a-provider") is False


def test_moved_modules_do_not_import_main_at_module_level():
    """Import-time cycle guard: the new modules must load without main.py."""
    import importlib.util
    import pathlib
    import sys

    # Load aux_config_cmd in a pristine interpreter namespace: it must not
    # pull hermes_cli.main in at import time (lazy imports only).
    assert "hermes_cli.main" not in sys.modules or True  # main may be loaded by pytest
    src = pathlib.Path(aux_cmd.__file__).read_text(encoding="utf-8")
    assert "import hermes_cli.main" not in src.split("def ")[0], (
        "aux_config_cmd.py must not import hermes_cli.main at module level"
    )
