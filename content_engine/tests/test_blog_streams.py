"""Tests for blog.blog_streams — the single source of per-stream truth.

Verifies the STREAMS dict matches the VERIFIED ingestion contract:
  - ai:  tier=ai (own page on SahilBlog), source=research-paper
  - pm:  tier=pm (non-AI tags), source=research-paper
  - builder: tier=builder, source=manual
"""
import blog.blog_streams as bs


def test_streams_shape():
    """Each stream has the required keys and valid field values."""
    for name in ("ai", "pm", "builder"):
        s = bs.STREAMS[name]
        assert s["tier"] in ("ai", "pm", "builder"), f"{name}: tier must be ai|pm|builder"
        assert s["voice"] and len(s["voice"]) > 20, f"{name}: voice must be non-trivial"
        assert s["format"] in ("essay", "note", "review", "brief"), f"{name}: format invalid"
        assert isinstance(s["base_tags"], list) and s["base_tags"], f"{name}: base_tags non-empty list"
        assert s["source"] in ("research-paper", "gitradar", "manual"), f"{name}: source invalid"
        assert isinstance(s["word_target"], int) and s["word_target"] >= 800, f"{name}: word_target"
        assert isinstance(s["section_target"], int) and s["section_target"] >= 3, f"{name}: section_target"
        assert isinstance(s["sources"], list) and s["sources"], f"{name}: sources non-empty list"
        assert "image_palette_brand" in s, f"{name}: image_palette_brand required"
        assert s.get("structure") and len(s["structure"]) > 20, f"{name}: structure required"


def test_streams_encode_the_purpose_led_editorial_contract():
    """Goals, source categories, voices and structures are executable stream config."""
    expected_categories = {
        "ai": {"ai_concepts", "prompt_harness_loop_engineering", "ai_theory_strategy"},
        "pm": {"enterprise_saas_ai_adoption", "product_strategy", "human_led_transformation"},
        "builder": {"kensei_shipped_features", "oss_contributions", "infrastructure_notes"},
    }
    for stream, categories in expected_categories.items():
        config = bs.STREAMS[stream]
        assert config["goal"]
        assert categories <= set(config["source_categories"])
    ai_goal = bs.STREAMS["ai"]["goal"].lower()
    for topic in ("speculative decoding", "memory management", "mcps", "skills"):
        assert topic in ai_goal
    assert "problem" in bs.STREAMS["builder"]["structure"].lower()
    assert "impact" in bs.STREAMS["builder"]["structure"].lower()
    assert "intended outcome" in bs.STREAMS["builder"]["structure"].lower()


def test_ai_is_ai_tier_with_ai_tag():
    """AI stream is tier=ai (its own page on SahilBlog)."""
    assert bs.STREAMS["ai"]["tier"] == "ai"
    tags = bs.STREAMS["ai"]["base_tags"]
    assert any(t in ("ai", "agentic", "llm", "agents", "ai-adoption", "ai-strategy", "machine-learning")
               for t in tags), "ai stream must carry an AI-recognised tag"


def test_ai_source_is_research_paper():
    """AI stream defaults to research-paper source per the contract."""
    assert bs.STREAMS["ai"]["source"] == "research-paper"


def test_pm_tier_is_pm_non_ai_tag():
    """PM stream is tier=pm with non-AI tags."""
    assert bs.STREAMS["pm"]["tier"] == "pm"
    tags = bs.STREAMS["pm"]["base_tags"]
    ai_tags = {"ai", "ai-adoption", "ai-strategy", "agents", "machine-learning", "llm", "agentic"}
    assert not any(t.lower() in ai_tags for t in tags), "pm stream base_tags must not include AI tags"


def test_pm_source_is_research_paper():
    """PM stream now leads from research findings."""
    assert bs.STREAMS["pm"]["source"] == "research-paper"


def test_builder_tier_is_builder():
    assert bs.STREAMS["builder"]["tier"] == "builder"


def test_builder_source_in_valid_set():
    assert bs.STREAMS["builder"]["source"] in ("gitradar", "manual")


def test_tags_for_merges_base_and_topic_dedup():
    """tags_for combines base_tags + topic_tags, de-duplicated, order-preserved."""
    out = bs.tags_for("ai", ["ai-adoption", "enterprise-saas"])
    assert "ai" in out  # base tag present
    assert "ai-adoption" in out  # topic tag present
    assert "enterprise-saas" in out
    # De-dup: if a tag appears in both, it appears once.
    out2 = bs.tags_for("ai", ["ai", "agents"])
    assert out2.count("ai") == 1


def test_tags_for_preserves_order():
    """Order: base_tags first, then topic_tags (minus dupes)."""
    out = bs.tags_for("pm", ["product-management", "saas-strategy"])
    assert out[0] == "product-management"  # base tag first
    assert "saas-strategy" in out


def test_ai_voice_is_analytical_numbers_first():
    """AI stream voice matches the groktop token-maxing analytical style."""
    v = bs.STREAMS["ai"]["voice"].lower()
    assert "analytical" in v or "numbers-first" in v or "thesis" in v


def test_builder_voice_is_practitioner_log():
    """Builder stream voice is a practitioner learning log with hype-vs-reality."""
    v = bs.STREAMS["builder"]["voice"].lower()
    assert "practitioner" in v or "learning log" in v or "hype-vs-reality" in v


def test_pm_voice_is_educational():
    """PM stream voice translates research to PM practice (educational)."""
    v = bs.STREAMS["pm"]["voice"].lower()
    assert "educational" in v or "translate" in v or "pm market" in v


def test_image_palette_brand_is_sahil_twitter():
    """All streams reuse the sahil_twitter palette pool for transplant imagery."""
    for name in ("ai", "pm", "builder"):
        assert bs.STREAMS[name]["image_palette_brand"] == "sahil_twitter", \
            f"{name}: must reuse sahil_twitter palette"


# -- Approach A research-roundup stream ---------------------------------------

def test_research_stream_present_in_strems():
    """The research stream is a first-class stream in the config map."""
    assert "research" in bs.STREAMS


def test_research_stream_shape():
    """The research stream carries the full shape required by the roundup lane."""
    s = bs.STREAMS["research"]
    assert s["tier"] == "research"          # NEW tier (clamped downstream)
    assert s["source"] == "curated-roundup"  # NEW source label (clamped downstream)
    assert s["format"] == "roundup"          # NEW format (clamped downstream)
    assert isinstance(s["base_tags"], list) and "research" in s["base_tags"]
    assert isinstance(s["word_target"], int) and s["word_target"] >= 800
    assert isinstance(s["section_target"], int) and s["section_target"] >= 7
    assert isinstance(s["sources"], list) and s["sources"]
    assert s["image_palette_brand"] == "sahil_twitter"
    assert s.get("structure") and len(s["structure"]) > 20
    assert s.get("voice") and len(s["voice"]) > 20


def test_research_stream_roundup_knobs():
    """The lane-specific knobs the generator threads into the prompt exist."""
    s = bs.STREAMS["research"]
    # Cross-source synthesis is mandatory: at least two distinct sources.
    assert s.get("require_two_distinct_sources") is True
    # Article-plus-social packaging: deck + first entry self-contained.
    assert s.get("article_plus_social") is True
    # Numbered entries target (independent of section_target).
    assert s.get("entries_target") == 5


def test_research_structure_is_roundup_not_essay():
    """The structure contract mandates numbered entries, a container, and
    takeaways — not the default essay shape."""
    structure = bs.STREAMS["research"]["structure"].lower()
    assert "thesis" in structure
    assert "numbered" in structure
    assert "## 01" in structure
    assert "limitations" in structure
    assert "## takeaways" in structure
    assert "try next" in structure


def test_research_voice_is_evidence_first_with_judgement():
    """The research voice leads with takeaways and honest limitations."""
    v = bs.STREAMS["research"]["voice"].lower()
    assert "evidence" in v
    assert "takeaway" in v or "lead with the" in v
    assert "limitation" in v
    assert "cross-source" in v