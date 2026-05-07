"""Tests for agent.codeact_promotion — the skill promotion pipeline.

Covers:
  - PromotionCandidate dataclass round-trip
  - build_skill_frontmatter() output format
  - write_promoted_skill() file I/O
  - extract_helper_functions() regex extraction
  - find_codeact_promotion_candidates() threshold logic
  - Pending promotions persistence (load/save/flag/remove)
"""

import json
import textwrap
from unittest.mock import patch, MagicMock

import pytest

from agent.codeact_promotion import (
    PromotionCandidate,
    build_skill_frontmatter,
    extract_helper_functions,
    find_codeact_promotion_candidates,
    flag_candidate,
    load_pending,
    remove_pending,
    save_pending,
    write_promoted_skill,
    _sanitize_fn_name,
    _PENDING_FILE_NAME,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_candidate():
    return PromotionCandidate(
        fn_name="fetch_weather",
        description="Fetch current weather for a city using wttr.in.",
        source_code='def fetch_weather(city: str) -> str:\n    """Fetch current weather for a city."""\n    import urllib.request\n    resp = urllib.request.urlopen(f"https://wttr.in/{city}?format=3")\n    return resp.read().decode()',
        domain="research",
        tags=["weather", "api"],
        session_id="abc123",
        occurrence_count=5,
        seen_in_sessions=["abc123", "def456"],
    )


@pytest.fixture
def sample_messages():
    """Messages containing repeated run_code calls with the same helper function."""
    code_block = textwrap.dedent("""\
        def scrape_prices(url: str) -> list:
            \"\"\"Extract prices from a product page.\"\"\"
            import re, urllib.request
            html = urllib.request.urlopen(url).read().decode()
            return re.findall(r'\\$[\\d,.]+', html)
    """)
    msg_template = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "run_code",
                    "arguments": json.dumps({"code": code_block}),
                }
            }
        ],
    }
    # 4 occurrences across the conversation
    return [msg_template for _ in range(4)]


# ---------------------------------------------------------------------------
# PromotionCandidate dataclass
# ---------------------------------------------------------------------------


class TestPromotionCandidate:
    def test_to_dict_roundtrip(self, sample_candidate):
        d = sample_candidate.to_dict()
        assert d["fn_name"] == "fetch_weather"
        assert d["occurrence_count"] == 5
        restored = PromotionCandidate.from_dict(d)
        assert restored.fn_name == sample_candidate.fn_name
        assert restored.source_code == sample_candidate.source_code
        assert restored.tags == sample_candidate.tags

    def test_defaults(self):
        c = PromotionCandidate(
            fn_name="f",
            description="d",
            source_code="x = 1",
        )
        assert c.domain == "general"
        assert c.tags == []
        assert c.session_id is None
        assert c.occurrence_count == 1
        assert c.seen_in_sessions == []
        assert c.promoted_at is None


# ---------------------------------------------------------------------------
# _sanitize_fn_name
# ---------------------------------------------------------------------------


class TestSanitizeFnName:
    def test_simple_name(self):
        assert _sanitize_fn_name("fetch_weather") == "fetch_weather"

    def test_dots_replaced(self):
        assert _sanitize_fn_name("my.func") == "my_func"

    def test_leading_digit_prefixed(self):
        assert _sanitize_fn_name("3things") == "fn_3things"

    def test_empty_name(self):
        assert _sanitize_fn_name("") == "fn_"

    def test_hyphens_preserved(self):
        assert _sanitize_fn_name("my-func") == "my-func"


# ---------------------------------------------------------------------------
# build_skill_frontmatter
# ---------------------------------------------------------------------------


class TestBuildSkillFrontmatter:
    def test_contains_required_fields(self, sample_candidate):
        fm = build_skill_frontmatter(sample_candidate)
        assert "---" in fm
        assert "name: fetch_weather" in fm
        assert "codeact_fn: fetch_weather" in fm
        assert "version: 1.0.0" in fm
        assert "author: CodeAct (auto-promoted)" in fm

    def test_tags_from_candidate(self, sample_candidate):
        fm = build_skill_frontmatter(sample_candidate)
        assert "weather" in fm
        assert "api" in fm

    def test_default_tags_when_empty(self):
        c = PromotionCandidate(
            fn_name="test_fn",
            description="Test",
            source_code="pass",
        )
        fm = build_skill_frontmatter(c)
        assert "codeact-promoted" in fm

    def test_session_id_recorded(self, sample_candidate):
        fm = build_skill_frontmatter(sample_candidate)
        assert "abc123" in fm


# ---------------------------------------------------------------------------
# write_promoted_skill
# ---------------------------------------------------------------------------


class TestWritePromotedSkill:
    def test_creates_skill_dir_and_md(self, sample_candidate, tmp_path):
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            result = write_promoted_skill(sample_candidate)

        assert result.exists()
        assert result.name == "SKILL.md"
        assert "promoted" in str(result)
        content = result.read_text(encoding="utf-8")
        assert "fetch_weather" in content
        assert "codeact_fn: fetch_weather" in content
        assert "def fetch_weather(city: str)" in content

    def test_sets_promoted_at_timestamp(self, sample_candidate, tmp_path):
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            write_promoted_skill(sample_candidate)
        assert sample_candidate.promoted_at is not None

    def test_skill_usage_bump_called(self, sample_candidate, tmp_path):
        with (
            patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path),
            patch("agent.codeact_promotion.logger"),
        ):
            with patch.dict(
                "sys.modules", {"tools": MagicMock(), "tools.skill_usage": MagicMock()}
            ):
                # Just verify no crash — bump_use is best-effort
                write_promoted_skill(sample_candidate)


# ---------------------------------------------------------------------------
# extract_helper_functions
# ---------------------------------------------------------------------------


class TestExtractHelperFunctions:
    def test_extracts_from_run_code_calls(self, sample_messages):
        found = extract_helper_functions(sample_messages)
        assert "scrape_prices" in found
        assert len(found["scrape_prices"]) == 4

    def test_ignores_non_run_code_calls(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "test"}),
                        }
                    }
                ],
            }
        ]
        assert extract_helper_functions(messages) == {}

    def test_handles_malformed_arguments(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": "not-valid-json",
                        }
                    }
                ],
            }
        ]
        assert extract_helper_functions(messages) == {}

    def test_handles_dict_arguments(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": {"code": "def hello():\n    pass\n"},
                        }
                    }
                ],
            }
        ]
        found = extract_helper_functions(messages)
        assert "hello" in found

    def test_multiple_functions_in_one_call(self):
        code = textwrap.dedent("""\
            def func_a():
                pass

            def func_b():
                pass
        """)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": json.dumps({"code": code}),
                        }
                    }
                ],
            }
        ]
        found = extract_helper_functions(messages)
        assert "func_a" in found
        assert "func_b" in found


# ---------------------------------------------------------------------------
# find_codeact_promotion_candidates
# ---------------------------------------------------------------------------


class TestFindCodeactPromotionCandidates:
    def test_below_threshold_not_included(self):
        code = "def helper():\n    pass\n"
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": json.dumps({"code": code}),
                        }
                    }
                ],
            }
        ]
        # Only 1 occurrence — below default min_occurrences=3
        candidates = find_codeact_promotion_candidates(messages)
        assert len(candidates) == 0

    def test_above_threshold_included(self, sample_messages):
        candidates = find_codeact_promotion_candidates(sample_messages)
        assert len(candidates) == 1
        assert candidates[0].fn_name == "scrape_prices"
        assert candidates[0].occurrence_count == 4

    def test_custom_min_occurrences(self, sample_messages):
        # With min_occurrences=5, our 4-occurrence function shouldn't qualify
        candidates = find_codeact_promotion_candidates(
            sample_messages, min_occurrences=5
        )
        assert len(candidates) == 0

    def test_docstring_extracted_as_description(self, sample_messages):
        candidates = find_codeact_promotion_candidates(sample_messages)
        assert "Extract prices" in candidates[0].description

    def test_no_docstring_uses_fallback(self):
        code = "def bare_func():\n    return 42\n"
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_code",
                            "arguments": json.dumps({"code": code}),
                        }
                    }
                ],
            }
            for _ in range(3)
        ]
        candidates = find_codeact_promotion_candidates(messages)
        assert len(candidates) == 1
        assert "bare_func" in candidates[0].description


# ---------------------------------------------------------------------------
# Pending promotions persistence
# ---------------------------------------------------------------------------


class TestPendingPersistence:
    def test_load_empty(self, tmp_path):
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            assert load_pending() == []

    def test_save_and_load(self, tmp_path):
        data = [{"fn_name": "test_fn", "description": "test", "source_code": "x = 1"}]
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            save_pending(data)
            loaded = load_pending()
        assert len(loaded) == 1
        assert loaded[0]["fn_name"] == "test_fn"

    def test_flag_candidate(self, sample_candidate, tmp_path):
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            flag_candidate(sample_candidate)
            pending = load_pending()
        assert len(pending) == 1
        assert pending[0]["fn_name"] == "fetch_weather"

    def test_flag_deduplicates(self, sample_candidate, tmp_path):
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            sample_candidate.occurrence_count = 3
            flag_candidate(sample_candidate)
            sample_candidate.occurrence_count = 7
            flag_candidate(sample_candidate)
            pending = load_pending()
        assert len(pending) == 1
        assert pending[0]["occurrence_count"] == 7  # latest wins

    def test_remove_pending(self, sample_candidate, tmp_path):
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            flag_candidate(sample_candidate)
            assert remove_pending("fetch_weather") is True
            assert load_pending() == []
            assert remove_pending("nonexistent") is False

    def test_corrupt_file_returns_empty(self, tmp_path):
        with patch("agent.codeact_promotion.get_hermes_home", return_value=tmp_path):
            p = tmp_path / "skills" / _PENDING_FILE_NAME
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("not json!!!")
            assert load_pending() == []
