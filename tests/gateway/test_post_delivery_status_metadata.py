from gateway.run import _post_delivery_status_metadata


def test_post_delivery_status_metadata_marks_notify_for_telegram():
    metadata = {"message_thread_id": 23}

    result = _post_delivery_status_metadata(metadata, platform="telegram")

    assert result == {"message_thread_id": 23, "notify": True}
    assert metadata == {"message_thread_id": 23}


def test_post_delivery_status_metadata_preserves_discord_non_conversational_marker():
    result = _post_delivery_status_metadata({"channel_id": "c1"}, platform="discord")

    assert result == {
        "channel_id": "c1",
        "non_conversational": True,
        "notify": True,
    }
