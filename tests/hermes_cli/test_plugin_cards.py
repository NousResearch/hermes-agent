from __future__ import annotations

from hermes_cli.plugin_cards import PluginCard, PluginCardAction, card_publisher_scope, present_plugin_card
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_plugin_card_action_accepts_unicode_label_and_registered_command_vocabulary():
    action = PluginCardAction("確認", "/release.review", "  exact args  ")

    assert action.id.startswith("action-")
    assert action.command == "release.review"
    assert action.args == "  exact args  "


def test_plugin_card_publication_is_nonblocking_and_surface_scoped():
    card = PluginCard(
        id="status",
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
            "id": "status",
            "plugin_id": "build-tools",
            "plugin_name": "Build Tools",
            "title": "Build ready",
            "body": "The candidate passed the local checks.",
            "actions": [{"id": "review", "label": "Review", "command": "build-review", "args": "candidate-7"}],
        }
    ]


def test_plugin_context_owns_card_attribution():
    ctx = PluginContext(PluginManifest(name="build-tools"), PluginManager())
    card = PluginCard(title="Ready", body="Review it")
    published = []

    assert ctx.publish_card(card) is False
    with card_publisher_scope(published.append):
        assert ctx.publish_card(card) is True

    assert published[0]["plugin_id"] == ctx.plugin_id
    assert published[0]["plugin_name"] == "build-tools"
