"""Regression tests for profile-scoped worker/host media aliases."""


def _turn(path):
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call", "function": {"name": "image_generate"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call",
            "content": '{"success": true, "image": "' + path + '"}',
        },
    ]


def test_worker_alias_is_not_reemitted_for_same_profile():
    from gateway.run import (
        _collect_auto_append_media_tags,
        _collect_history_media_paths,
    )

    history_paths = _collect_history_media_paths(
        _turn(
            "/host-hermes/profiles/profile-a/cache/images/old.png"
        ),
    )
    tags, _ = _collect_auto_append_media_tags(
        _turn("/worker-home/.hermes/cache/images/old.png"),
        history_offset=9999,
        history_media_paths=history_paths,
        profile_name="profile-a",
    )
    assert tags == []


def test_same_basename_in_another_profile_is_not_suppressed():
    from gateway.run import (
        _collect_auto_append_media_tags,
        _collect_history_media_paths,
    )

    history_paths = _collect_history_media_paths(
        _turn(
            "/host-hermes/profiles/profile-b/cache/images/shared.png"
        ),
    )
    tags, _ = _collect_auto_append_media_tags(
        _turn("/worker-home/.hermes/cache/images/shared.png"),
        history_offset=9999,
        history_media_paths=history_paths,
        profile_name="profile-a",
    )
    assert tags == ["MEDIA:/worker-home/.hermes/cache/images/shared.png"]
