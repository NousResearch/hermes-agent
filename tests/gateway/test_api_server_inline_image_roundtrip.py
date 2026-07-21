"""Round-trip regression for inline MEDIA image data URLs (api_server).

``_resolve_media_to_data_urls`` inlines ``MEDIA:`` tags as ``![image](data:...)``
markdown in outbound assistant text (up to a ~7 MB base64 payload). When a
stateless OpenAI-compatible client echoes that assistant turn back in
``messages[]``, ``_normalize_multimodal_content`` used to slice the whole string
at ``MAX_NORMALIZED_TEXT_LENGTH``, cutting the data URL mid-base64 and dropping
any assistant text after the image. The legacy ``_normalize_chat_content`` had
the same bare slices on its string and list paths. These tests pin that the
surrounding text survives while the oversized payload is elided.
"""

from gateway.platforms.api_server import (
    _normalize_chat_content,
    _normalize_multimodal_content,
    MAX_NORMALIZED_TEXT_LENGTH,
)


def _big_data_url(total_chars: int = MAX_NORMALIZED_TEXT_LENGTH * 2) -> str:
    payload = "A" * total_chars
    return f"data:image/png;base64,{payload}"


def test_assistant_text_after_inline_image_survives_string_roundtrip():
    prefix = "Here is the chart you asked for:\n\n"
    tail = "\n\nThe chart shows a 20 percent increase in Q3. SUFFIX_MARKER_KEEP"
    content = f"{prefix}![image]({_big_data_url()})" + tail
    assert len(content) > MAX_NORMALIZED_TEXT_LENGTH

    result = _normalize_multimodal_content(content)

    assert isinstance(result, str)
    assert "SUFFIX_MARKER_KEEP" in result
    assert "AAAAAAAAAA" not in result
    assert len(result) <= MAX_NORMALIZED_TEXT_LENGTH


def test_assistant_text_after_inline_image_survives_list_text_part():
    tail = " ... SUFFIX_MARKER_KEEP"
    text = f"before ![image]({_big_data_url()}) after{tail}"
    content = [{"type": "text", "text": text}]

    result = _normalize_multimodal_content(content)

    joined = result if isinstance(result, str) else "\n".join(
        p.get("text", "") for p in result if isinstance(p, dict)
    )
    assert "SUFFIX_MARKER_KEEP" in joined
    assert "AAAAAAAAAA" not in joined


def test_structured_image_url_part_is_untouched():
    tiny = "data:image/png;base64,iVBORw0KGgo="
    content = [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": tiny}},
    ]
    result = _normalize_multimodal_content(content)
    assert isinstance(result, list)
    urls = [
        p["image_url"]["url"]
        for p in result
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    assert tiny in urls


def test_plain_oversized_text_still_capped():
    content = "B" * (MAX_NORMALIZED_TEXT_LENGTH * 2)
    result = _normalize_multimodal_content(content)
    assert isinstance(result, str)
    assert len(result) == MAX_NORMALIZED_TEXT_LENGTH


def test_short_text_passthrough():
    content = "just a short normal message"
    assert _normalize_multimodal_content(content) == content


def test_assistant_text_after_inline_image_survives_chat_content_string():
    tail = "\n\nThe chart shows a 20 percent increase in Q3. SUFFIX_MARKER_KEEP"
    content = f"Here is the chart:\n\n![image]({_big_data_url()})" + tail
    assert len(content) > MAX_NORMALIZED_TEXT_LENGTH

    result = _normalize_chat_content(content)

    assert "SUFFIX_MARKER_KEEP" in result
    assert "AAAAAAAAAA" not in result
    assert len(result) <= MAX_NORMALIZED_TEXT_LENGTH


def test_assistant_text_after_inline_image_survives_chat_content_list_items():
    str_item = f"str item ![image]({_big_data_url()}) STR_MARKER_KEEP"
    dict_item = {"type": "text", "text": f"dict item ![image]({_big_data_url()}) DICT_MARKER_KEEP"}

    result = _normalize_chat_content([str_item, dict_item])

    assert "STR_MARKER_KEEP" in result
    assert "DICT_MARKER_KEEP" in result
    assert "AAAAAAAAAA" not in result
