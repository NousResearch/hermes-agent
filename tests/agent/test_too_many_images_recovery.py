"""Per-request image COUNT ceilings: a count 400 is learned per route and never re-hit.

Port of MiniMax-AI/minimax-code#431; salvage of #104914.
"""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error
from agent.image_count_limit import apply_learned_image_count_limit, learn_image_count_limit


class _ApiError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 400
        self.body = {"error": {"message": message}}
        self.response = None


def _img(tag):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tag}"}}


def _conversation(n_old_tool_images):
    """Older tool screenshots, then the user's upload, then a tool round AFTER it (the user row is not last)."""
    rows = [{"role": "system", "content": "sys"}]
    for i in range(n_old_tool_images):
        rows += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
             "function": {"name": "browser_vision", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": [{"type": "text", "text": "shot"}, _img(f"t{i}")]},
        ]
    rows.append({"role": "user", "content": [{"type": "text", "text": "what is in this?"}, _img("upload")]})
    rows += [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "cz", "type": "function",
         "function": {"name": "browser_vision", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "cz", "content": [{"type": "text", "text": "shot"}, _img("newest")]},
    ]
    return rows


def _image_urls(rows):
    out = []
    for m in rows:
        if isinstance(m.get("content"), list):
            out += [p["image_url"]["url"].rsplit(",", 1)[1] for p in m["content"] if p.get("type") == "image_url"]
    return out


def test_count_rejection_is_learned_and_keeps_the_latest_upload():
    agent = SimpleNamespace(provider="deepinfra", model="vlm", _image_count_limits={})
    history = _conversation(10)  # 12 images; the route accepts 8
    err = _ApiError("Too many images in request: 12 > 8")

    attempt = list(history)
    removed, limit = learn_image_count_limit(agent, err, attempt)
    assert (removed, limit) == (len(_image_urls(history)) - len(_image_urls(attempt)), 8)
    kept = _image_urls(attempt)
    assert len(kept) <= 8 and "upload" in kept and "newest" in kept and "t0" not in kept
    assert len(_image_urls(history)) == 12  # history keeps every pixel

    # Next turn on the same route: capped before sending, without another 400.
    next_turn = _conversation(14)
    apply_learned_image_count_limit(agent, next_turn)
    assert len(_image_urls(next_turn)) <= 8 and "upload" in _image_urls(next_turn)
    # Another model on the same provider is untouched.
    other = SimpleNamespace(provider="deepinfra", model="other", _image_count_limits=agent._image_count_limits)
    untouched = _conversation(14)
    assert apply_learned_image_count_limit(other, untouched) == 0


def test_hosted_count_rejections_classify_terminal_not_generic_retry():
    for text in (
        "Too many images in request: 31 > 30",
        "Too many images were provided, we currently limit the number of images per conversation to 60",
        "Exceeded limit on max data-uri per request: 250",
        "At most 2 image(s) may be provided in one prompt.",
    ):
        verdict = classify_api_error(_ApiError(text), provider="custom")
        assert verdict.reason == FailoverReason.too_many_images, text
        assert verdict.retryable is False, text  # the strip recovery runs first; an unchanged resend can't pass
    assert classify_api_error(_ApiError("image exceeds 5 MB maximum"), provider="custom").reason == (
        FailoverReason.image_too_large
    )
