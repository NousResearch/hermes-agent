"""Retire stale vision_analyze / screenshot tool images inside the protected tail.

Issue #92699: native embeds sitting in protect_last_n ride every later
request, compression savings stay tiny, and anti-thrash disables further
compaction. Pass 2 of ``_prune_old_tool_results`` only strips images
*outside* the tail; this suite pins the extra keep-newest pass.
"""

from __future__ import annotations

from agent.context_compressor import (
    ContextCompressor,
    _MAX_KEEP_TOOL_IMAGES,
    _content_has_images,
    _tool_content_has_images,
)


def _compressor() -> ContextCompressor:
    c = ContextCompressor.__new__(ContextCompressor)
    c.quiet_mode = True
    return c


def _image_tool(i: int, *, blob: str = "A" * 400) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "vision_analyze",
                        "arguments": f'{{"image_url":"shot{i}.png"}}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": [
                {"type": "text", "text": f"Image attached natively shot {i}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{blob}{i}"},
                },
            ],
        },
    ]


def _image_bearing_tools(messages: list[dict]) -> list[dict]:
    return [
        m
        for m in messages
        if m.get("role") == "tool" and _tool_content_has_images(m.get("content"))
    ]


class TestRetireStaleToolImagesInProtectedTail:
    def test_keep_newest_images_inside_a_large_protect_window(self):
        """Five native embeds all fit in protect_last_n=20; only newest N stay."""
        msgs: list[dict] = [{"role": "user", "content": "screenshot QA"}]
        for i in range(5):
            msgs.extend(_image_tool(i))
        msgs.append({"role": "user", "content": "compare the last few"})

        out, pruned = _compressor()._prune_old_tool_results(
            msgs, protect_tail_count=20,
        )
        assert pruned >= 5 - _MAX_KEEP_TOOL_IMAGES
        kept = _image_bearing_tools(out)
        assert len(kept) == _MAX_KEEP_TOOL_IMAGES
        kept_ids = [m["tool_call_id"] for m in kept]
        assert kept_ids == [f"call_{i}" for i in range(5 - _MAX_KEEP_TOOL_IMAGES, 5)]

    def test_multimodal_envelopes_outside_keep_window_become_text(self):
        msgs: list[dict] = [{"role": "user", "content": "go"}]
        for i in range(_MAX_KEEP_TOOL_IMAGES + 1):
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"env_{i}",
                    "type": "function",
                    "function": {"name": "vision_analyze", "arguments": "{}"},
                }],
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": f"env_{i}",
                "content": {
                    "_multimodal": True,
                    "content": [
                        {"type": "text", "text": f"shot {i}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,XYZ{i}"},
                        },
                    ],
                    "text_summary": f"native shot {i}",
                },
            })

        out, _ = _compressor()._prune_old_tool_results(
            msgs, protect_tail_count=20,
        )
        oldest = next(m for m in out if m.get("tool_call_id") == "env_0")
        assert isinstance(oldest["content"], str)
        assert "screenshot removed" in oldest["content"]
        assert "native shot 0" in oldest["content"]
        newest = next(
            m for m in out
            if m.get("tool_call_id") == f"env_{_MAX_KEEP_TOOL_IMAGES}"
        )
        assert _tool_content_has_images(newest["content"])

    def test_user_uploads_are_not_retired(self):
        user_img = {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,USERUPLOAD"},
                },
            ],
        }
        msgs = [user_img]
        for i in range(_MAX_KEEP_TOOL_IMAGES + 2):
            msgs.extend(_image_tool(i))

        out, _ = _compressor()._prune_old_tool_results(
            msgs, protect_tail_count=20,
        )
        assert _content_has_images(out[0]["content"])
        assert out[0]["content"][1]["image_url"]["url"].endswith("USERUPLOAD")


class TestSharedImageStripHelper:
    """One strip policy for pass 3.5 and the demote pass (#92783 follow-up)."""

    def test_demote_pass_drops_stale_api_content_on_image_strip(self):
        """Pass 2's image demotion must drop the api_content sidecar.

        Before the shared _strip_images_from_tool_msg helper, only pass 3.5
        dropped the sidecar; the demote branches left it behind, letting
        replay restore pre-strip bytes.
        """
        from agent.context_compressor import _strip_images_from_tool_msg

        msg = {
            "role": "tool",
            "tool_call_id": "c1",
            "api_content": "stale exact-wire copy with image bytes",
            "content": [
                {"type": "text", "text": "shot"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + "A" * 400},
                },
            ],
        }
        new_msg = _strip_images_from_tool_msg(msg)
        assert new_msg is not None
        assert "api_content" not in new_msg
        # Input untouched (copy-on-write).
        assert "api_content" in msg
        assert _content_has_images(msg["content"])
        assert not _content_has_images(new_msg["content"])

    def test_envelope_collapses_to_summary_string(self):
        from agent.context_compressor import _strip_images_from_tool_msg

        msg = {
            "role": "tool",
            "tool_call_id": "c2",
            "api_content": "stale",
            "content": {
                "_multimodal": True,
                "content": [
                    {"type": "text", "text": "s"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,XYZ"},
                    },
                ],
                "text_summary": "native shot",
            },
        }
        new_msg = _strip_images_from_tool_msg(msg)
        assert new_msg is not None
        assert isinstance(new_msg["content"], str)
        assert "screenshot removed" in new_msg["content"]
        assert "native shot" in new_msg["content"]
        assert "api_content" not in new_msg

    def test_imageless_content_returns_none(self):
        from agent.context_compressor import _strip_images_from_tool_msg

        msg = {"role": "tool", "tool_call_id": "c3", "content": "plain text"}
        assert _strip_images_from_tool_msg(msg) is None
        msg2 = {
            "role": "tool",
            "tool_call_id": "c4",
            "content": [{"type": "text", "text": "no images here"}],
        }
        assert _strip_images_from_tool_msg(msg2) is None
