"""Chat startup must explain a mixed-version ImportError (#96900).

A half-updated install can ship a newer ``cli.py`` that imports
``resolve_turn_limit`` / ``split_model_config_default`` from
``hermes_cli.config`` while the older ``config.py`` does not export them.
Construction of ``HermesCLI`` then dies before the agent-setup mixin can
print ``partial_update_hint``. ``cmd_chat`` is the load-bearing catch:
bare ``hermes`` and ``hermes chat`` (including the fast-chat launch path)
all go through it.
"""

from argparse import Namespace
import io
import sys
import types

import pytest

from hermes_constants import emit_partial_update_hint, partial_update_hint


def _chat_args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "no_restore_cwd": False,
        "toolsets": None,
        "skills": None,
        "tui": False,
        "tui_dev": False,
        "cli": True,
        "verbose": None,
        "quiet": True,
        "query": "hello",
        "image": None,
        "worktree": False,
        "checkpoints": False,
        "pass_session_id": False,
        "max_turns": None,
        "ignore_rules": False,
        "ignore_user_config": False,
        "safe_mode": False,
        "compact": False,
        "source": None,
        "yolo": False,
        "accept_hooks": False,
    }
    base.update(overrides)
    return Namespace(**base)


def _missing_config_name_error(name: str = "resolve_turn_limit") -> ImportError:
    exc = ImportError(
        f"cannot import name '{name}' from 'hermes_cli.config'"
    )
    exc.name = "hermes_cli.config"
    return exc


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(mod, "_resolve_session_by_name_or_id", lambda val: val)
    return mod


def test_emit_hint_for_missing_resolve_turn_limit():
    exc = _missing_config_name_error("resolve_turn_limit")
    buf = io.StringIO()

    assert emit_partial_update_hint(exc, file=buf) is True
    text = buf.getvalue()
    assert "resolve_turn_limit" in text
    assert "hermes update" in text
    assert "partially-updated" in text


def test_emit_hint_for_missing_split_model_config_default():
    exc = _missing_config_name_error("split_model_config_default")
    assert partial_update_hint(exc)
    buf = io.StringIO()
    assert emit_partial_update_hint(exc, file=buf) is True
    assert "hermes update" in buf.getvalue()


def test_emit_hint_stays_silent_for_third_party_import_error():
    exc = ImportError("cannot import name 'dumps' from 'requests'")
    exc.name = "requests"
    buf = io.StringIO()
    assert emit_partial_update_hint(exc, file=buf) is False
    assert buf.getvalue() == ""


@pytest.mark.parametrize(
    "name",
    ["resolve_turn_limit", "split_model_config_default"],
)
def test_cmd_chat_prints_update_hint_when_config_helper_is_missing(
    main_mod, monkeypatch, capsys, name
):
    def boom(**_kwargs):
        raise _missing_config_name_error(name)

    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(main=boom))

    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_chat(_chat_args())

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert name in err
    assert "hermes update" in err
    assert "partially-updated" in err


def test_cmd_chat_still_reraises_unrelated_import_errors(main_mod, monkeypatch):
    exc = ImportError("cannot import name 'dumps' from 'requests'")
    exc.name = "requests"

    def boom(**_kwargs):
        raise exc

    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(main=boom))

    with pytest.raises(ImportError) as excinfo:
        main_mod.cmd_chat(_chat_args())

    assert excinfo.value is exc
