"""Tests for the LLM chain and the topic-leak sanitiser
(no raw git commit subjects in topics/titles)."""
import llm_generate as lg
import topics as tp


# ── LLM chain ──────────────────────────────────────────────────────────────

def test_fallback_chain_is_ollama_cloud_only():
    bases = [(c["base"], c["model"], c["provider"]) for c in lg._FREE_FALLBACK_CHAIN]
    assert bases[0] == ("https://ollama.com/v1", "deepseek-v4-flash", "ollama")
    assert bases[1] == ("https://ollama.com/v1", "gpt-oss:120b", "ollama")
    assert not any("opencode" in c["base"] for c in lg._FREE_FALLBACK_CHAIN)


def test_key_for_provider(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-xyz")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-abc")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-123")
    assert lg._key_for("ollama") == "ollama-xyz"
    # Legacy OpenCode key resolution is retained for explicit fallback entries/operator overrides.
    assert lg._key_for("opencode") == "go-abc"
    assert lg._key_for("gemini") == "gemini-123"


def test_llm_configs_attaches_ollama_keys(monkeypatch):
    monkeypatch.delenv("CONTENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CONTENT_LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-xyz")
    cfgs = lg._llm_configs()
    assert cfgs[0]["base"] == "https://ollama.com/v1"
    assert cfgs[0]["model"] == "deepseek-v4-flash"
    assert all(c["key"] == "ollama-xyz" for c in cfgs)


def test_longform_chain_has_provider_fallbacks(monkeypatch):
    monkeypatch.delenv("CONTENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CONTENT_LLM_MODEL", raising=False)
    cfgs = lg._llm_configs(longform=True)
    models = [c["model"] for c in cfgs]
    assert models[0] == "deepseek/deepseek-v4-flash"
    assert "deepseek-v4-flash" in models
    assert "minimax-m3" in models
    assert "gemini-2.5-flash" in models


def test_short_and_longform_differ(monkeypatch):
    monkeypatch.delenv("CONTENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CONTENT_LLM_MODEL", raising=False)
    short = [c["model"] for c in lg._llm_configs(longform=False)]
    long = [c["model"] for c in lg._llm_configs(longform=True)]
    assert short[0] == "deepseek-v4-flash"
    assert long[0] == "deepseek/deepseek-v4-flash"


def test_fabricated_numbers_catches_growth_metrics():
    import article_gates as ag
    assert ag._fabricated_numbers("Just hit 1,000 users this week", "")
    assert ag._fabricated_numbers("200 downloads in a day", "")
    assert ag._fabricated_numbers("conversion rose 45%", "")
    assert not ag._fabricated_numbers("I cut 3 features in 2 weeks", "")
    assert not ag._fabricated_numbers("we hit 1,000 users", "the launch reached 1,000 users")


def test_env_override_is_tried_first(monkeypatch):
    monkeypatch.setenv("CONTENT_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("CONTENT_LLM_MODEL", "custom-model")
    cfgs = lg._llm_configs()
    assert (cfgs[0]["base"], cfgs[0]["model"]) == ("https://example.test/v1", "custom-model")
    assert any(c["model"] == "deepseek-v4-flash" for c in cfgs)
    assert any(c["model"] == "gpt-oss:120b" for c in cfgs)


def test_gemini_override_uses_gemini_key(monkeypatch):
    monkeypatch.setenv("CONTENT_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    monkeypatch.setenv("CONTENT_LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-xyz")
    cfgs = lg._llm_configs(longform=True)
    assert cfgs[0]["base"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert cfgs[0]["model"] == "gemini-2.5-flash"
    assert cfgs[0]["key"] == "gemini-xyz"


# ── Topic-leak sanitiser ───────────────────────────────────────────────────

def test_strips_conventional_commit_prefix():
    assert tp._clean_topic_summary(
        "feat(content): non-infographic scene transplant") == "non-infographic scene transplant"
    assert tp._clean_topic_summary("fix: gateway crash loop") == "gateway crash loop"
    assert tp._clean_topic_summary("refactor(api)!: drop v1") == "drop v1"


def test_strips_trailer():
    assert tp._clean_topic_summary(
        "feat: add thing\n\nCo-Authored-By: X <x@y.z>") == "add thing"


def test_merge_commit_yields_empty():
    assert tp._clean_topic_summary("Merge branch 'main' into feature") == ""


def test_clean_passes_through_normal_text():
    assert tp._clean_topic_summary("How I cut agent context 99x") == "How I cut agent context 99x"


def test_no_commit_prefix_in_built_topic():
    cleaned = tp._clean_topic_summary("chore(deps): bump three to 0.184")
    assert not cleaned.lower().startswith("chore")
    assert cleaned == "bump three to 0.184"
