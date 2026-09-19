from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from hermes_cli.plugin_cards import PluginCard, PluginCardAction, card_publisher_scope, present_plugin_card
from hermes_cli.cli_modal_mixin import CLIModalMixin
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_plugin_card_action_accepts_unicode_label_and_registered_command_vocabulary():
    action = PluginCardAction("確認", "/release.review", "  exact args  ")

    assert action.command == "release.review"
    assert action.args == "  exact args  "


def test_plugin_card_publication_is_nonblocking_and_surface_scoped():
    card = PluginCard(
        title="Build ready",
        body="The candidate passed the local checks.",
        actions=(PluginCardAction("Review", "build-review", "candidate-7"),),
    )

    assert card.text_fallback() == "Build ready\n\nThe candidate passed the local checks."
    assert present_plugin_card("build-tools", "Build Tools", card) is False

    published = []
    with card_publisher_scope(published.append):
        assert present_plugin_card("build-tools", "Build Tools", card) is True
    with card_publisher_scope(lambda _payload: False):
        assert present_plugin_card("build-tools", "Build Tools", card) is False

    assert published == [
        {
            "plugin_id": "build-tools",
            "plugin_name": "Build Tools",
            "title": "Build ready",
            "body": "The candidate passed the local checks.",
            "actions": [{"label": "Review", "command": "build-review", "args": "candidate-7"}],
        }
    ]


def test_plugin_card_actions_need_no_host_identity_or_unique_labels():
    card = PluginCard(
        title="Team skill available",
        body="Choose the exact action to take.",
        actions=(
            PluginCardAction("Share", "skill-choice", "read"),
            PluginCardAction("Share", "skill-choice", "inspect"),
        ),
    )

    assert [action.as_dict() for action in card.actions] == [
        {"label": "Share", "command": "skill-choice", "args": "read"},
        {"label": "Share", "command": "skill-choice", "args": "inspect"},
    ]

    with pytest.raises(ValueError, match="at least one action"):
        PluginCard(title="Not actionable", body="No choices", actions=())


def test_plugin_context_owns_card_attribution():
    ctx = PluginContext(PluginManifest(name="build-tools"), PluginManager())
    card = PluginCard(
        title="Ready", body="Review it", actions=(PluginCardAction("Review in Portal", "inspect", ""),)
    )
    published = []

    assert ctx.publish_card(card) is False
    with card_publisher_scope(published.append):
        assert ctx.publish_card(card) is True

    assert published[0]["plugin_id"] == ctx.plugin_id
    assert published[0]["plugin_name"] == "build-tools"


def test_classic_notice_selects_registered_owning_action_and_replaces_with_next_notice():
    manager = PluginManager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name="team-tools"), manager)
    seen = []
    next_card = PluginCard(
        title="Update available",
        body="Choose the next action.",
        actions=(PluginCardAction("Review in Portal", "skill-finish", "exact next args"),),
    )
    ctx.register_command("skill-choice", lambda args: seen.append(args) or next_card)
    ctx.register_command("skill-finish", lambda args: seen.append(args) or "Done.")

    class Surface(CLIModalMixin):
        _app = object()
        _approval_state = None
        _clarify_state = None
        _secret_state = None
        _sudo_state = None
        _slash_confirm_state = None

        def __init__(self):
            self.prompts = []

        def _prompt_text_input_modal(self, **kwargs):
            self.prompts.append(kwargs)
            return kwargs["choices"][0][0]

    surface = Surface()
    assert surface._present_plugin_card("team-tools", "Team Tools", PluginCard(
        title="Team skill available",
        body="A new team skill is available.",
        actions=(PluginCardAction("Review in Portal", "skill-choice", "exact args"),),
    ), manager=manager) is True

    assert seen == ["exact args", "exact next args"]
    assert [prompt["title"] for prompt in surface.prompts] == [
        "Team Tools · Team skill available",
        "Team Tools · Update available",
    ]


def test_classic_notice_uses_neutral_prompt_marker_without_weakening_confirmations():
    from cli import HermesCLI

    class Surface(CLIModalMixin):
        _app = object()
        _approval_state = None
        _clarify_state = None
        _secret_state = None
        _sudo_state = None
        _slash_confirm_state = None

        def _prompt_text_input_modal(self, **kwargs):
            self.prompt = kwargs
            return None

    surface = Surface()
    card = PluginCard(
        title="Team update",
        body="A routine plugin notice.",
        actions=(PluginCardAction("Review", "skill-review"),),
    )

    assert surface._present_plugin_card("team-tools", "Team Tools", card) is True
    assert surface.prompt["warning"] is False

    cli = HermesCLI.__new__(HermesCLI)
    cli._voice_recording = cli._voice_processing = cli._voice_mode = False
    cli._sudo_state = cli._secret_state = cli._approval_state = None
    cli._clarify_state = None
    cli._clarify_freetext = cli._command_running = cli._agent_running = False
    with patch.object(HermesCLI, "_get_tui_terminal_width", return_value=100):
        setattr(cli, "_slash_confirm_state", {"warning": False})
        assert "".join(text for _style, text in cli._get_tui_prompt_fragments()).startswith("ℹ")

        setattr(cli, "_slash_confirm_state", {"title": "Destructive confirmation"})
        assert "".join(text for _style, text in cli._get_tui_prompt_fragments()).startswith("⚠")

        setattr(cli, "_slash_confirm_state", {"warning": False})
        setattr(cli, "_approval_state", {"command": "needs approval"})
        assert "".join(text for _style, text in cli._get_tui_prompt_fragments()).startswith("⚠")


def test_classic_notice_is_readable_but_noninteractive_without_an_app(capsys):
    class Headless(CLIModalMixin):
        _app = None

    card = PluginCard(
        title="Qualified skill",
        body="This body remains readable without terminal interaction.",
        actions=(PluginCardAction("Review in Portal", "skill-choice", "read"),),
    )

    assert Headless()._present_plugin_card("team-tools", "Team Tools", card) is False
    assert "Qualified skill\n\nThis body remains readable" in capsys.readouterr().out


def test_classic_notice_shows_action_failure_and_keeps_the_menu(capsys):
    manager = PluginManager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name="team-tools"), manager)

    def fail(_args):
        raise RuntimeError("Deliberate fixture failure")

    ctx.register_command("skill-fail", fail)

    class Surface(CLIModalMixin):
        _app = object()
        _approval_state = None
        _clarify_state = None
        _secret_state = None
        _sudo_state = None
        _slash_confirm_state = None

        def __init__(self):
            self.selections = iter(("0", None))

        def _prompt_text_input_modal(self, **_kwargs):
            return next(self.selections)

    card = PluginCard(
        title="Exact review",
        body="Review before confirming.",
        actions=(PluginCardAction("Show failure", "skill-fail", "fail"),),
    )

    assert Surface()._present_plugin_card("team-tools", "Team Tools", card, manager=manager) is True
    assert "Action failed: Deliberate fixture failure" in capsys.readouterr().out


def test_classic_notice_does_not_replace_an_existing_human_prompt(capsys):
    class Surface(CLIModalMixin):
        _app = object()
        _approval_state = None
        _clarify_state = None
        _command_palette_state = object()
        _secret_state = None
        _sudo_state = None
        _slash_confirm_state = None

        def _prompt_text_input_modal(self, **_kwargs):
            raise AssertionError("notice replaced the active prompt")

    card = PluginCard(
        title="Team update",
        body="A team update arrived while another choice was active.",
        actions=(PluginCardAction("Review in Portal", "skill-choice", "read"),),
    )

    assert Surface()._present_plugin_card("team-tools", "Team Tools", card) is False
    assert "Team update" in capsys.readouterr().out


def test_classic_command_publication_keeps_its_originating_session(monkeypatch):
    from contextvars import copy_context
    from types import SimpleNamespace
    from cli import HermesCLI

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="team-tools"), manager)
    contexts = []
    ctx.register_command("capture", lambda _args: contexts.append(copy_context()))
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_commands", lambda: manager._plugin_commands)

    class Surface(CLIModalMixin):
        _app = SimpleNamespace(is_running=True)
        session_id = "session-a"
        _plugin_notice_lock = threading.Lock()

        def _present_plugin_card(self, *_args, **_kwargs):
            return True

    surface = Surface()
    HermesCLI._run_plugin_slash_command(surface, "/capture", "")
    surface.session_id = "session-b"
    card = PluginCard("Old session notice", "Must not move to the new session.", (PluginCardAction("Review", "capture"),))
    assert contexts[0].run(ctx.publish_card, card) is False


def test_classic_publication_returns_while_notice_waits_for_a_choice(capsys):
    prompt_started = threading.Event()
    release_prompt = threading.Event()

    class App:
        is_running = True

    class Surface(CLIModalMixin):
        _app = App()
        _approval_state = None
        _clarify_state = None
        _command_palette_state = None
        _model_picker_state = None
        _secret_state = None
        _sudo_state = None
        _slash_confirm_state = None
        _should_exit = False
        session_id = "session-a"

        def __init__(self):
            self._plugin_notice_lock = threading.Lock()

        def _prompt_text_input_modal(self, **_kwargs):
            prompt_started.set()
            release_prompt.wait(timeout=5)
            return None

    surface = Surface()
    payload = {
        "plugin_id": "team-tools",
        "plugin_name": "Team Tools",
        "title": "Team update",
        "body": "Review this update.",
        "actions": [{"label": "Review", "command": "skill-review", "args": "read"}],
    }
    returned = threading.Event()
    accepted = []

    caller = threading.Thread(
        target=lambda: (accepted.append(surface._plugin_card_publisher()(payload)), returned.set()),
        daemon=True,
    )
    caller.start()
    try:
        assert returned.wait(timeout=2), "publication blocked on the user's choice"
        assert accepted == [True]
        assert prompt_started.wait(timeout=2)
        assert surface._plugin_notice_lock.locked()
        assert surface._plugin_card_publisher()(payload) is False
    finally:
        release_prompt.set()
        caller.join(timeout=2)

    assert surface._plugin_notice_lock.acquire(timeout=2)
    surface._plugin_notice_lock.release()
    assert capsys.readouterr().out.count("Review this update.") == 1


@pytest.mark.parametrize(
    ("obsolete_field", "obsolete_value"),
    (("session_id", "session-b"), ("_secret_state", object()), ("_should_exit", True)),
)
def test_classic_publication_drops_obsolete_or_blocked_action(obsolete_field, obsolete_value):
    manager = PluginManager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name="team-tools"), manager)
    actions = []
    ctx.register_command("skill-review", lambda args: actions.append(args) or "Done.")
    prompt_started = threading.Event()
    release_prompt = threading.Event()

    class App:
        is_running = True

    class Surface(CLIModalMixin):
        _app = App()
        _approval_state = None
        _clarify_state = None
        _command_palette_state = None
        _model_picker_state = None
        _secret_state = None
        _sudo_state = None
        _slash_confirm_state = None
        _should_exit = False
        session_id = "session-a"

        def __init__(self):
            self._plugin_notice_lock = threading.Lock()

        def _prompt_text_input_modal(self, **_kwargs):
            prompt_started.set()
            release_prompt.wait(timeout=5)
            return "0"

    surface = Surface()
    payload = {
        "plugin_id": ctx.plugin_id,
        "plugin_name": "Team Tools",
        "title": "Team update",
        "body": "Review this update.",
        "actions": [{"label": "Review", "command": "skill-review", "args": "exact args"}],
    }

    assert surface._plugin_card_publisher()(payload, manager=manager) is True
    assert prompt_started.wait(timeout=2)
    setattr(surface, obsolete_field, obsolete_value)
    release_prompt.set()

    assert surface._plugin_notice_lock.acquire(timeout=2)
    surface._plugin_notice_lock.release()
    assert actions == []
