import pytest
from unittest.mock import patch, MagicMock

from agent.codeact_skill_injector import (
    SkillNamespaceInjector,
    _get_codeact_fn,
    _resolve_collisions,
)
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_skill_utils():
    with patch("agent.codeact_skill_injector.get_all_skill_frontmatters") as mock:
        yield mock


@pytest.fixture
def mock_recently_used():
    with patch("agent.codeact_skill_injector._load_recently_used_skills") as mock:
        mock.return_value = []
        yield mock


def _make_registry_with_tools(*tool_names):
    """Create a mock registry whose _snapshot_entries returns entries with the given names."""
    registry = MagicMock(spec=ToolRegistry)
    entries = []
    for name in tool_names:
        entry = MagicMock()
        entry.name = name
        entry.schema = {"name": name}
        entries.append(entry)
    registry._snapshot_entries.return_value = entries
    return registry


# ---------------------------------------------------------------------------
# Basic stub generation
# ---------------------------------------------------------------------------


class TestSkillStubGeneration:
    def test_empty_injector(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {}
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        result = injector.get_skill_stubs()
        assert result == ""

    def test_functional_skill_injection(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {
            "myskill": {
                "name": "myskill",
                "description": "Does something cool.",
                "codeact_fn": "my_skill_func",
            }
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        result = injector.get_skill_stubs()
        assert "def my_skill_func(" in result
        assert '"""Does something cool."""' in result
        assert "_call_tool('__skill__'" in result
        assert '"skill_name": "myskill"' in result

    def test_skills_without_codeact_fn_ignored(
        self, mock_skill_utils, mock_recently_used
    ):
        mock_skill_utils.return_value = {
            "plain_skill": {
                "name": "plain_skill",
                "description": "I have no codeact_fn.",
            },
            "callable_skill": {
                "name": "callable_skill",
                "description": "I do have one.",
                "codeact_fn": "callable_fn",
            },
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        result = injector.get_skill_stubs()
        assert "callable_fn" in result
        assert "plain_skill" not in result

    def test_codeact_fn_true_uses_skill_name(
        self, mock_skill_utils, mock_recently_used
    ):
        mock_skill_utils.return_value = {
            "my_research_tool": {
                "name": "my_research_tool",
                "description": "Research tool.",
                "codeact_fn": True,
            }
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        names = injector.get_skill_names()
        assert "my_research_tool" in names

    def test_description_escaping(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {
            "escape_skill": {
                "name": "escape_skill",
                "description": 'Has """triple quotes""" and \\ backslashes',
                "codeact_fn": "escape_fn",
            }
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        result = injector.get_skill_stubs()
        # Should not have unescaped triple-quotes breaking the Python source
        assert "def escape_fn(" in result


# ---------------------------------------------------------------------------
# Collision resolution
# ---------------------------------------------------------------------------


class TestCollisionResolution:
    def test_tool_collision_renames_skill(self):
        candidates = [("my_skill", "web_search", {"description": "x"})]
        tool_names = {"web_search", "read_file", "help"}
        result = _resolve_collisions(candidates, tool_names)
        assert len(result) == 1
        assert result[0][1] == "skill_web_search"

    def test_no_collision_preserves_name(self):
        candidates = [("my_skill", "unique_name", {"description": "x"})]
        tool_names = {"web_search", "help"}
        result = _resolve_collisions(candidates, tool_names)
        assert result[0][1] == "unique_name"

    def test_duplicate_skill_names_deduplicated(self):
        candidates = [
            ("skill_a", "shared_fn", {"description": "a"}),
            ("skill_b", "shared_fn", {"description": "b"}),
        ]
        tool_names = set()
        result = _resolve_collisions(candidates, tool_names)
        assert len(result) == 1
        assert result[0][0] == "skill_a"  # first wins

    def test_collision_with_builtins(self):
        """_resolve_collisions detects collisions when builtin names are in tool_names."""
        candidates = [
            ("bad_name", "help", {"description": "collision"}),
            ("good_name", "real_fn", {"description": "ok"}),
        ]
        # Built-in kernel function names must be included in tool_names for
        # _resolve_collisions to detect them.  The full select_skills() path
        # does this via _collect_tool_names(); here we simulate it.
        tool_names = {
            "web_search",
            "help",
            "promote_to_skill",
            "_call_tool",
            "__protected__",
        }
        result = _resolve_collisions(candidates, tool_names)
        # "help" collides with built-in → renamed to skill_help
        assert result[0][1] == "skill_help"
        assert result[1][1] == "real_fn"


# ---------------------------------------------------------------------------
# Selection priority
# ---------------------------------------------------------------------------


class TestSelectionPriority:
    def test_explicitly_loaded_skills_first(self, mock_skill_utils, mock_recently_used):
        mock_recently_used.return_value = []
        mock_skill_utils.return_value = {
            "alpha": {"name": "alpha", "description": "a", "codeact_fn": "alpha_fn"},
            "beta": {"name": "beta", "description": "b", "codeact_fn": "beta_fn"},
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        result = injector.select_skills(explicitly_loaded={"beta"})
        # beta should be first (explicitly loaded)
        assert result[0][0] == "beta"

    def test_recently_used_skills_second_tier(
        self, mock_skill_utils, mock_recently_used
    ):
        mock_recently_used.return_value = ["gamma"]
        mock_skill_utils.return_value = {
            "alpha": {"name": "alpha", "description": "a", "codeact_fn": "alpha_fn"},
            "gamma": {"name": "gamma", "description": "g", "codeact_fn": "gamma_fn"},
            "delta": {"name": "delta", "description": "d", "codeact_fn": "delta_fn"},
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        result = injector.select_skills()
        names = [r[0] for r in result]
        # gamma (recently used) should come before delta (random)
        assert names.index("gamma") < names.index("delta")

    def test_max_skills_truncation(self, mock_skill_utils, mock_recently_used):
        mock_recently_used.return_value = []
        mock_skill_utils.return_value = {
            f"skill_{i}": {
                "name": f"skill_{i}",
                "description": "x",
                "codeact_fn": f"fn_{i}",
            }
            for i in range(25)
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry, max_skills=10)
        result = injector.select_skills()
        assert len(result) == 10

    def test_max_skills_zero_injects_nothing(
        self, mock_skill_utils, mock_recently_used
    ):
        mock_recently_used.return_value = []
        mock_skill_utils.return_value = {
            "skill_a": {"name": "skill_a", "description": "a", "codeact_fn": "fn_a"},
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry, max_skills=0)
        result = injector.select_skills()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Help registry
# ---------------------------------------------------------------------------


class TestSkillHelpRegistry:
    def test_help_registry_contains_skills(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {
            "research_tool": {
                "name": "research_tool",
                "description": "Search arXiv papers by keyword.",
                "codeact_fn": "arxiv_search",
            }
        }
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        help_reg = injector.get_skill_help_registry()
        assert "arxiv_search" in help_reg
        compact, full = help_reg["arxiv_search"]
        assert "arxiv_search(**kwargs)" in compact
        assert "Search arXiv papers" in compact
        assert "research_tool" in full

    def test_empty_skills_empty_registry(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {}
        registry = ToolRegistry()
        injector = SkillNamespaceInjector(registry)
        help_reg = injector.get_skill_help_registry()
        assert help_reg == {}


# ---------------------------------------------------------------------------
# _get_codeact_fn helper
# ---------------------------------------------------------------------------


class TestGetCodeactFn:
    def test_string_value(self):
        assert _get_codeact_fn({"codeact_fn": "my_fn"}) == "my_fn"

    def test_true_value_uses_name(self):
        assert _get_codeact_fn({"codeact_fn": True, "name": "skill_x"}) == "skill_x"

    def test_false_value_returns_none(self):
        assert _get_codeact_fn({"codeact_fn": False}) is None

    def test_empty_string_returns_none(self):
        assert _get_codeact_fn({"codeact_fn": ""}) is None

    def test_whitespace_string_returns_none(self):
        assert _get_codeact_fn({"codeact_fn": "  "}) is None

    def test_missing_key_returns_none(self):
        assert _get_codeact_fn({"description": "no codeact_fn"}) is None


# ---------------------------------------------------------------------------
# Namespace integration — __protected__ includes skill names
# ---------------------------------------------------------------------------


class TestNamespaceSkillProtection:
    """Verify that build_tool_namespace_source adds skill function names to __protected__."""

    def test_protected_includes_skill_names(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {
            "test_skill": {
                "name": "test_skill",
                "description": "A test skill.",
                "codeact_fn": "test_skill_fn",
            }
        }
        from agent.codeact_namespace import build_tool_namespace_source

        registry = MagicMock(spec=ToolRegistry)
        registry._snapshot_entries.return_value = []

        injector = SkillNamespaceInjector(registry)
        source = build_tool_namespace_source(
            registry=registry,
            skill_injector=injector,
        )
        # The __protected__ list in the source should include the skill function name
        assert "test_skill_fn" in source
        assert "__protected__" in source

    def test_skill_stubs_in_output(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {
            "my_skill": {
                "name": "my_skill",
                "description": "Does stuff.",
                "codeact_fn": "do_stuff",
            }
        }
        from agent.codeact_namespace import build_tool_namespace_source

        registry = MagicMock(spec=ToolRegistry)
        registry._snapshot_entries.return_value = []

        injector = SkillNamespaceInjector(registry)
        source = build_tool_namespace_source(
            registry=registry,
            skill_injector=injector,
        )
        assert "def do_stuff(**kwargs):" in source
        assert "Callable Skills" in source

    def test_no_injector_no_skills(self, mock_skill_utils, mock_recently_used):
        from agent.codeact_namespace import build_tool_namespace_source

        registry = MagicMock(spec=ToolRegistry)
        registry._snapshot_entries.return_value = []

        source = build_tool_namespace_source(
            registry=registry,
            skill_injector=None,
        )
        assert "Callable Skills" not in source

    def test_help_registry_includes_skills(self, mock_skill_utils, mock_recently_used):
        mock_skill_utils.return_value = {
            "arxiv": {
                "name": "arxiv",
                "description": "Search arXiv papers.",
                "codeact_fn": "arxiv_search",
            }
        }
        from agent.codeact_namespace import build_tool_namespace_source

        registry = MagicMock(spec=ToolRegistry)
        registry._snapshot_entries.return_value = []

        injector = SkillNamespaceInjector(registry)
        source = build_tool_namespace_source(
            registry=registry,
            skill_injector=injector,
        )
        # _HELP_REGISTRY in the source should contain the skill
        assert "arxiv_search" in source


# ---------------------------------------------------------------------------
# Soft reset preserves skills
# ---------------------------------------------------------------------------


class TestSoftResetPreservesSkills:
    """Integration: kernel soft_reset should keep skill functions alive."""

    def test_protected_set_contains_skills(self, mock_skill_utils, mock_recently_used):
        """Verify that the __protected__ list in the generated source includes
        skill function names, which means soft_reset will preserve them."""
        mock_skill_utils.return_value = {
            "persistent_skill": {
                "name": "persistent_skill",
                "description": "Should survive reset.",
                "codeact_fn": "persistent_fn",
            }
        }
        from agent.codeact_namespace import build_tool_namespace_source
        import json
        import re

        registry = MagicMock(spec=ToolRegistry)
        registry._snapshot_entries.return_value = []

        injector = SkillNamespaceInjector(registry)
        source = build_tool_namespace_source(
            registry=registry,
            skill_injector=injector,
        )

        # Extract the __protected__ assignment from the generated source
        match = re.search(r"__protected__\s*=\s*(\[.*?\])", source)
        assert match is not None, "__protected__ assignment not found in source"
        protected_list = json.loads(match.group(1))
        assert "persistent_fn" in protected_list
