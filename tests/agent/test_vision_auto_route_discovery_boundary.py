"""#123998: the auxiliary *vision* auto-route must honor the same discovery boundary as the text
route. Once a main provider is selected, an unavailable main must not silently widen to guessing
another logged-in provider (e.g. OpenRouter via ``OPENROUTER_API_KEY``) and ship the user's image
there. With no main selected, the built-in discovery chain still runs.
"""

import json

import pytest


_CONFIG = """model:
  default: gpt-5.5
  provider: openai-codex
  base_url: https://chatgpt.com/backend-api/codex
"""

_NO_MAIN_CONFIG = """model:
  default: ""
  provider: ""
"""

# A *selected* main that is text-only / vision-blind (kimi-coding #17076). Unlike an unavailable
# vision-capable main, this main was never going to serve the image, so the boundary must NOT fire:
# the vision route falls through to the aggregator chain exactly as before (#123998 regression).
_TEXT_ONLY_MAIN_CONFIG = """model:
  default: kimi-code
  provider: kimi-coding
"""


def _trap(tmp_path, monkeypatch, config: str):
    h = tmp_path / "home"
    h.mkdir()
    (h / "config.yaml").write_text(config)
    (h / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}})
    )
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-repro")
    from agent import auxiliary_client as ac

    built: list[str] = []

    class _Fake:
        def __init__(self, base_url):
            self.base_url = base_url
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise RuntimeError("request would be sent to " + str(self.base_url))

    def _factory(*, api_key, base_url, **kw):
        built.append(str(base_url))
        return _Fake(base_url)

    monkeypatch.setattr(ac, "_create_openai_client", _factory)
    ac._client_cache.clear()
    return ac, built


_IMAGE = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    }
]


def test_text_task_refuses_to_guess_openrouter(tmp_path, monkeypatch):
    """Baseline: the text route already refuses (this must stay green)."""
    ac, built = _trap(tmp_path, monkeypatch, _CONFIG)
    with pytest.raises(ac.AuxiliaryClientUnavailable):
        ac.call_llm(task="compression", messages=[{"role": "user", "content": "x"}], max_tokens=5)
    assert not any("openrouter" in u for u in built)


def test_vision_task_also_refuses_to_guess_openrouter(tmp_path, monkeypatch):
    """#123998: the vision route must match the text route and refuse to guess OpenRouter."""
    ac, built = _trap(tmp_path, monkeypatch, _CONFIG)
    with pytest.raises(ac.AuxiliaryClientUnavailable):
        ac.call_llm(task="vision", messages=_IMAGE, max_tokens=5)
    assert not any("openrouter" in u for u in built), (
        f"vision request leaked to a non-selected provider: {built}"
    )


def test_vision_text_only_main_still_falls_through_to_openrouter(tmp_path, monkeypatch):
    """#123998 regression: a *text-only* selected main (kimi-coding, no vision support) was never
    going to serve the image, so the discovery boundary must NOT gate it — the vision route falls
    through to the aggregator chain (OpenRouter) as it did before the boundary was added (#17076,
    #50426). The three tests above all use a vision-capable main, so none of them cover this."""
    ac, built = _trap(tmp_path, monkeypatch, _TEXT_ONLY_MAIN_CONFIG)
    with pytest.raises(RuntimeError, match="openrouter.ai"):
        ac.call_llm(task="vision", messages=_IMAGE, max_tokens=5)
    assert any("openrouter" in u for u in built), (
        f"text-only main was wrongly gated instead of falling through to the aggregator: {built}"
    )


def test_vision_discovery_still_runs_with_no_main_selected(tmp_path, monkeypatch):
    """The boundary only applies once a main provider is chosen: a fresh install (no main) still
    reaches the built-in discovery chain (here OpenRouter, which the trap turns into a send)."""
    ac, built = _trap(tmp_path, monkeypatch, _NO_MAIN_CONFIG)
    with pytest.raises(RuntimeError, match="openrouter.ai"):
        ac.call_llm(task="vision", messages=_IMAGE, max_tokens=5)
    assert any("openrouter" in u for u in built)
