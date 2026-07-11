"""Support Ops validation must never infer meaning from authored text."""

from gateway.support_ops_routing import (
    lint_and_resolve_discord_content,
    lint_discord_target_for_content,
    lint_discord_thread_create_target,
)
import gateway.support_ops_team_registry as registry
from gateway.support_ops_team_registry import SKYVISION_BACKEND_CHANNEL_ID


def test_free_text_is_opaque_and_unchanged():
    samples = (
        "Моля пиши на Алекс за voucher backend казуса.",
        "Alex, send this to Emil in control tower",
        "Клиентът каза: „Пламена ми обеща“.",
        "@unknown-user е raw evidence, не route instruction.",
    )
    for text in samples:
        result = lint_and_resolve_discord_content(text)
        assert result.ok is True
        assert result.content == text


def test_authored_mentions_never_force_or_block_a_lane():
    result = lint_discord_target_for_content(
        "<@1282940511962791959> моля провери.",
        chat_id="some-model-selected-public-channel",
    )
    assert result.ok is True
    assert result.expected_channel_id is None


def test_unstructured_title_and_starter_never_infer_target_person():
    result = lint_discord_thread_create_target(
        "Алекс: провери ваучера",
        channel_id="model-selected-public-channel",
        initial_message="Моля пиши на Алекс.",
    )
    assert result.ok is True


def test_explicit_unknown_target_requires_model_clarification():
    result = lint_discord_thread_create_target(
        "Нов казус",
        channel_id="model-selected-public-channel",
        initial_message="context",
        target_person="ivan_h",
    )
    assert result.ok is False
    assert result.blocked_reason == "blocked_unknown_target_person_requires_clarification"


def test_explicit_resolved_target_is_validated_mechanically():
    result = lint_discord_thread_create_target(
        "Нов казус",
        channel_id=SKYVISION_BACKEND_CHANNEL_ID,
        initial_message="context",
        target_person="alex",
    )
    assert result.ok is True


def test_registry_exposes_no_free_text_inference_helpers():
    assert not hasattr(registry, "infer_requested_person_phrase")
    assert not hasattr(registry, "infer_salutation_team_member")
    assert not hasattr(registry, "infer_team_members_from_text")


def test_model_confirmed_alias_is_persisted_and_resolved_exactly(tmp_path, monkeypatch):
    alias_path = tmp_path / "aliases.json"
    monkeypatch.setattr(registry, "_learned_alias_path", lambda: alias_path)
    learned = registry.learn_team_member_alias("Алекс БЕ", "alex")
    assert learned == {"alias": "алекс бе", "member_key": "alex"}
    resolution = registry.resolve_team_member("Алекс БЕ")
    assert resolution.status == "resolved"
    assert resolution.member.key == "alex"
