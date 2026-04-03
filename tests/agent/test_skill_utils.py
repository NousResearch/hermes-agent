import pytest

from agent.skill_utils import extract_skill_conditions


class TestExtractSkillConditions:
    def test_non_mapping_metadata_does_not_crash(self):
        frontmatter = {"metadata": "definitely not a dict"}

        assert extract_skill_conditions(frontmatter) == {
            "fallback_for_toolsets": [],
            "requires_toolsets": [],
            "fallback_for_tools": [],
            "requires_tools": [],
        }

    def test_json_string_metadata_is_parsed_defensively(self):
        frontmatter = {
            "metadata": '{"hermes":{"requires_toolsets":["web"],"requires_tools":["browser_navigate"]}}'
        }

        assert extract_skill_conditions(frontmatter) == {
            "fallback_for_toolsets": [],
            "requires_toolsets": ["web"],
            "fallback_for_tools": [],
            "requires_tools": ["browser_navigate"],
        }

    def test_json_string_hermes_block_is_parsed_defensively(self):
        frontmatter = {
            "metadata": {
                "hermes": '{"fallback_for_toolsets":["web"],"fallback_for_tools":["web_search"]}'
            }
        }

        assert extract_skill_conditions(frontmatter) == {
            "fallback_for_toolsets": ["web"],
            "requires_toolsets": [],
            "fallback_for_tools": ["web_search"],
            "requires_tools": [],
        }
