"""Per-image token cost learned from provider usage (agent/image_token_cost.py, #70328).

A flat per-image constant undercounts multimodal local models 2-4x (a GUI loop then hits
provider 400s before compaction can fire) and overcounts providers that downscale. The
provider prices every image exactly on the request that carries it, so the residual between
the real prompt count and ``anchor + text-only delta`` teaches the per-image cost.
"""

from types import SimpleNamespace

from agent import image_token_cost as itc
from agent.model_metadata import estimate_messages_tokens_rough
from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor


def _img():
    return {"role": "user", "content": [{"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 4000}}]}


def _agent(anchor):
    return SimpleNamespace(_usage_anchor=anchor, model="vision-local", base_url="http://127.0.0.1:8080/v1")


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(itc, "_LEARNED", {})
    monkeypatch.setattr(itc, "_LOADED_SCOPES", {itc._scope()})
    monkeypatch.setattr(itc, "_cache_path", lambda: tmp_path / "image_token_costs.json")


def test_residual_on_an_image_delta_teaches_the_per_image_cost(monkeypatch, tmp_path):
    """Two screenshots appended since the last real reading; the provider reports 8,000 tokens
    more than the text-only projection -> 4,000 per image, and every estimator prices images at
    that from now on (trigger and budget walk read the same bound value)."""
    _isolate(monkeypatch, tmp_path)
    history = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "ok"}]
    anchor = capture_usage_anchor(10_000, 5, history)
    history += [_img(), {"role": "assistant", "content": "looking"}, _img()]
    agent = _agent(anchor)
    with itc.image_cost_context(0):
        text_only = anchored_context_tokens(history, anchor)
    with itc.image_cost_context(None):
        learned = itc.calibrate_from_usage(agent, history, text_only + 2 * 4_000)
        assert learned == 4_000
        assert itc.current_image_token_cost() == 4_000
        assert estimate_messages_tokens_rough([_img()]) >= 4_000
    # Persisted per model@host: a fresh process starts calibrated.
    monkeypatch.setattr(itc, "_LEARNED", {})
    monkeypatch.setattr(itc, "_LOADED_SCOPES", set())
    assert itc.learned_image_token_cost("vision-local", "http://127.0.0.1:8080/v1") == 4_000
    assert itc.learned_image_token_cost("other-model", "http://127.0.0.1:8080/v1") == itc.DEFAULT_IMAGE_TOKEN_COST


def test_nothing_learned_without_images_or_anchor(monkeypatch, tmp_path):
    """Text-only deltas, missing anchors and implausible residuals teach nothing: a text-estimate
    error must never be mistaken for an image price."""
    _isolate(monkeypatch, tmp_path)
    history = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "ok"}]
    anchor = capture_usage_anchor(10_000, 5, history)
    history += [{"role": "user", "content": "no image here"}]
    assert itc.calibrate_from_usage(_agent(anchor), history, 50_000) is None
    history += [_img()]
    assert itc.calibrate_from_usage(_agent(None), history, 50_000) is None
    assert itc.calibrate_from_usage(_agent(anchor), history, 10_001) is None  # residual < plausible floor
    assert itc._LEARNED == {}


def _learn_in(home, tokens_per_image):
    """Calibrate one image's cost while the given profile's home override is active."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        history = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "ok"}]
        anchor = capture_usage_anchor(10_000, 5, history)
        history += [_img(), {"role": "assistant", "content": "looking"}]
        agent = _agent(anchor)
        with itc.image_cost_context(0):
            text_only = anchored_context_tokens(history, anchor)
        with itc.image_cost_context(None):
            return itc.calibrate_from_usage(agent, history, text_only + tokens_per_image)
    finally:
        reset_hermes_home_override(token)


def _current_in(home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        return itc.learned_image_token_cost("vision-local", "http://127.0.0.1:8080/v1")
    finally:
        reset_hermes_home_override(token)


def test_learned_costs_are_profile_scoped(monkeypatch, tmp_path):
    """A process serving two profiles in the same process (desktop multiplex) must not let one
    profile's learned per-image cost bleed into another's, even though both loads happen off the
    same in-memory module -- and switching back to a profile already seen this process must still
    see its own learned value, not reload/lose it (#70328 follow-up)."""
    monkeypatch.setattr(itc, "_LEARNED", {})
    monkeypatch.setattr(itc, "_LOADED_SCOPES", set())

    prof_a = tmp_path / "profA"
    prof_b = tmp_path / "profB"
    for p in (prof_a, prof_b):
        (p / "cache").mkdir(parents=True)

    assert _learn_in(prof_a, 4_000) == 4_000
    # Profile B, touched for the first time in the SAME process, must not inherit A's value.
    assert _current_in(prof_b) == itc.DEFAULT_IMAGE_TOKEN_COST
    assert _learn_in(prof_b, 900) == 900
    # Switching back to A must still see A's own learned value, unaffected by B's calibration.
    assert _current_in(prof_a) == 4_000

    # Each profile's on-disk file holds only its own learned data -- no cross-write.
    import json

    a_file = json.loads((prof_a / "cache" / "image_token_costs.json").read_text())
    b_file = json.loads((prof_b / "cache" / "image_token_costs.json").read_text())
    assert list(a_file.values()) == [4_000]
    assert list(b_file.values()) == [900]
