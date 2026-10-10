"""A stored ``nous`` tool selection pins that tool to the Nous Tool Gateway.

Under that selection the runtime ignores vendor credentials, BYO plugin providers and local
defaults, and reports an unavailable gateway through ``selection_error``. The setup summary and
the shared feature state must agree: with no gateway, every Nous-manageable tool is reported
unavailable, and its hint never names a vendor credential the selection would ignore.
"""

import pytest

from hermes_cli import nous_subscription, setup_summary
from hermes_cli.config import load_config, save_config

_ROWS = {
    "web": setup_summary._web_row,
    "browser": setup_summary._browser_row,
    "image_gen": setup_summary._image_gen_row,
    "video_gen": setup_summary._video_gen_row,
    "tts": setup_summary._tts_row,
    "stt": setup_summary._stt_row,
}

# Every credential or local endpoint a row, a plugin probe or the feature resolver could read.
# All are set, and all are ignored while the tool's stored selection is ``nous``.
_VENDOR_ENV = (
    "EXA_API_KEY", "PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "TAVILY_API_KEY",
    "PERPLEXITY_API_KEY", "KEENABLE_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY", "BROWSER_USE_API_KEY",
    "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID", "FAL_KEY", "OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY",
    "ELEVENLABS_API_KEY", "MINIMAX_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
    "DEEPINFRA_API_KEY",
)


@pytest.mark.parametrize(
    ("key", "section_field"), sorted(nous_subscription._GATEWAY_SECTION_FIELDS.items()),
    ids=sorted(nous_subscription._GATEWAY_SECTION_FIELDS),
)
def test_unavailable_nous_selection_is_unavailable_and_never_asks_for_an_ignored_key(monkeypatch, key, section_field):
    for name in _VENDOR_ENV:
        monkeypatch.setenv(name, "synthetic")
    # Local engines are installed too, so only the gateway is missing.
    monkeypatch.setattr("hermes_cli.setup._module_installed", lambda module: True)
    monkeypatch.setattr(nous_subscription, "_has_agent_browser", lambda: True)
    monkeypatch.setattr(nous_subscription, "_local_browser_runnable", lambda: True)
    # No Nous account, so the Tool Gateway is unavailable.
    monkeypatch.setattr(nous_subscription, "get_nous_portal_account_info", lambda **_: None)
    section, field = section_field
    config = load_config()
    config[section] = {**(config.get(section) or {}), field: "nous"}
    save_config(config)  # setup saves before the summary; plugin probes read the file
    config = load_config()

    features = nous_subscription.get_nous_subscription_features(config)
    name, available, hint = _ROWS[key](config, features)

    assert features.features[key].available is False  # the state `hermes status` reads
    assert available is False, name
    assert "hermes tools" in hint
    assert [env for env in _VENDOR_ENV if env in hint] == []
