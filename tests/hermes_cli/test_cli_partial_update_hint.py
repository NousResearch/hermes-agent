"""Chat startup must explain a mixed-version ImportError (#96900).

A half-updated install can ship a newer launcher importing config helpers
that the older ``hermes_cli.config`` does not export. The transport launch
must preserve the diagnostic previously provided around ``cli.main``.
``cmd_chat`` is the load-bearing catch:
bare ``hermes`` and ``hermes chat`` (including the fast-chat launch path)
all go through it.
"""

from argparse import Namespace
import io

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
@pytest.mark.parametrize("entry", ["cmd_chat", "_run_oneshot_from_args"])
def test_cmd_chat_prints_update_hint_when_config_helper_is_missing(
    main_mod, monkeypatch, capsys, name, entry
):
    def boom(_args):
        raise _missing_config_name_error(name)

    monkeypatch.setattr("hermes_cli.gateway_chat.launch_from_args", boom)

    with pytest.raises(SystemExit) as excinfo:
        getattr(main_mod, entry)(_chat_args())

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert name in err
    assert "hermes update" in err
    assert "partially-updated" in err


@pytest.mark.parametrize("entry", ["cmd_chat", "_run_oneshot_from_args"])
def test_cmd_chat_still_reraises_unrelated_import_errors(main_mod, monkeypatch, entry):
    exc = ImportError("cannot import name 'dumps' from 'requests'")
    exc.name = "requests"

    def boom(_args):
        raise exc

    monkeypatch.setattr("hermes_cli.gateway_chat.launch_from_args", boom)

    with pytest.raises(ImportError) as excinfo:
        getattr(main_mod, entry)(_chat_args())

    assert excinfo.value is exc
