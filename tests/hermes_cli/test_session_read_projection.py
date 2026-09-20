from hermes_cli.web_routers.sessions import _without_inline_images


def test_without_inline_images_preserves_remote_refs_and_does_not_mutate_input():
    original = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,large"}},
        {"type": "image_url", "image_url": "https://example.test/image.png"},
        {"type": "text", "text": "caption"},
    ]

    projected = _without_inline_images(original)

    assert projected[0]["image_url"]["url"] == "[image]"
    assert projected[1]["image_url"] == "https://example.test/image.png"
    assert projected[2]["text"] == "caption"
    assert original[0]["image_url"]["url"].startswith("data:image/")
