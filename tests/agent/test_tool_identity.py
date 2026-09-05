"""Canonical tool identity across runtimes.

The display layer keys every curated table on *native* tool names, so a
child runtime speaking its own vocabulary (Claude Agent SDK ``Bash``,
Codex ``exec_command``) used to miss every lookup and render as a bare
``Bash: "..."`` line. These tests pin the property that matters: the same
logical operation must produce the same display output on every lane.

Parity is asserted between lanes rather than against hardcoded strings, so
the tests keep their meaning when a verb or preview budget is retuned. A
few anchors assert real values as well, so the suite still fails if the
verb tables were emptied entirely.
"""

import pytest

from agent.display import (
    build_status_phrase,
    build_tool_preview,
    get_tool_emoji,
    get_tool_verb,
    tool_verb_connector,
    verb_drops_preview,
)
from agent.tool_identity import (
    canonical_tool_args,
    canonical_tool_name,
    canonicalize_tool_call,
    strip_mcp_namespace,
)

# (label, tool_name, args) triples for one logical operation per lane.
TERMINAL_LANES = [
    ("native", "terminal", {"command": "ls -la /tmp"}),
    ("claude-agent-sdk", "Bash", {"command": "ls -la /tmp"}),
    ("codex", "exec_command", {"cmd": "ls -la /tmp"}),
]

READ_LANES = [
    ("native", "read_file", {"path": "/tmp/example.py"}),
    ("claude-agent-sdk", "Read", {"file_path": "/tmp/example.py"}),
    ("hermes-tools-mcp", "mcp__hermes-tools__read_file", {"path": "/tmp/example.py"}),
]

PATCH_LANES = [
    ("native", "patch", {"path": "/tmp/example.py"}),
    ("claude-agent-sdk", "Edit", {"file_path": "/tmp/example.py"}),
    (
        "codex",
        "apply_patch",
        {"changes": [{"kind": "update", "path": "/tmp/example.py"}]},
    ),
]

TODO_LANES = [
    ("native", "todo_list", {"todos": [{"content": "Ship it"}]}),
    ("claude-agent-sdk", "TodoWrite", {"todos": [{"content": "Ship it"}]}),
    ("legacy", "todo", {"todos": [{"content": "Ship it"}]}),
]


class TestCanonicalToolName:
    """Foreign names map onto native ones; everything else is left alone."""

    @pytest.mark.parametrize(
        "foreign,native",
        [
            ("Bash", "terminal"),
            ("exec_command", "terminal"),
            ("Read", "read_file"),
            ("Write", "write_file"),
            ("Edit", "patch"),
            ("MultiEdit", "patch"),
            ("apply_patch", "patch"),
            ("Grep", "search_files"),
            ("Glob", "search_files"),
            ("Task", "delegate_task"),
            ("TodoWrite", "todo_list"),
            ("todo", "todo_list"),
            ("cronjob", "cronjob_manage"),
            ("process", "process_manage"),
            ("tour", "gui_tour"),
            ("tip", "show_tip"),
        ],
    )
    def test_foreign_names_map_to_native(self, foreign, native):
        assert canonical_tool_name(foreign) == native

    @pytest.mark.parametrize(
        "native", ["terminal", "read_file", "patch", "todo_list"]
    )
    def test_native_names_are_unchanged(self, native):
        assert canonical_tool_name(native) == native

    @pytest.mark.parametrize(
        "unmapped", ["ToolSearch", "BashOutput", "KillShell", "ExitPlanMode"]
    )
    def test_tools_without_an_honest_counterpart_keep_their_name(self, unmapped):
        """Better to render an unknown name than to force a wrong verb."""
        assert canonical_tool_name(unmapped) == unmapped

    def test_empty_name_is_returned_unchanged(self):
        assert canonical_tool_name("") == ""


class TestMcpNamespace:
    """Hermes' own tools lose the namespace; third-party servers keep it."""

    @pytest.mark.parametrize(
        "namespaced",
        ["mcp__hermes-tools__read_file", "mcp.hermes-tools.read_file"],
    )
    def test_internal_server_is_stripped(self, namespaced):
        assert strip_mcp_namespace(namespaced) == "read_file"
        assert canonical_tool_name(namespaced) == "read_file"

    @pytest.mark.parametrize(
        "third_party",
        ["mcp__linear__get_issue", "mcp.linear.get_issue"],
    )
    def test_third_party_servers_keep_their_namespace(self, third_party):
        """Hiding which server a tool came from would be misleading."""
        assert strip_mcp_namespace(third_party) == third_party
        assert canonical_tool_name(third_party) == third_party

    @pytest.mark.parametrize(
        "malformed",
        ["mcp__hermes-tools__", "mcp__hermes-tools", "mcp__", "mcp."],
    )
    def test_malformed_namespaces_are_left_alone(self, malformed):
        assert strip_mcp_namespace(malformed) == malformed


class TestCanonicalToolArgs:
    """Argument keys are mapped per-tool, never globally."""

    def test_foreign_keys_are_mapped(self):
        assert canonical_tool_args("Read", {"file_path": "/tmp/a"})["path"] == "/tmp/a"
        assert canonical_tool_args("exec_command", {"cmd": "ls"})["command"] == "ls"

    def test_native_key_wins_over_alias(self):
        args = {"path": "/native", "file_path": "/foreign"}
        assert canonical_tool_args("Read", args)["path"] == "/native"

    def test_input_dict_is_never_mutated(self):
        args = {"file_path": "/tmp/a"}
        canonical_tool_args("Read", args)
        assert args == {"file_path": "/tmp/a"}

    def test_native_case_returns_the_same_object(self):
        """The common path must not allocate a copy."""
        args = {"path": "/tmp/a"}
        assert canonical_tool_args("read_file", args) is args

    def test_non_dict_args_yield_an_empty_dict(self):
        for bad in (None, "string", 42, ["list"]):
            assert canonical_tool_args("Read", bad) == {}

    @pytest.mark.parametrize("empty", [None, "", [], {}])
    def test_empty_alias_values_are_skipped(self, empty):
        """An empty foreign value must not shadow a missing native key."""
        assert "path" not in canonical_tool_args("Read", {"file_path": empty})

    def test_alias_scoping_does_not_corrupt_other_tools(self):
        """``query`` means different things per tool, so aliases are per-tool."""
        args = canonical_tool_args("web_search", {"query": "solar"})
        assert args == {"query": "solar"}

    def test_codex_patch_changes_list_yields_a_path(self):
        args = canonical_tool_args(
            "apply_patch", {"changes": [{"kind": "update", "path": "/tmp/a.py"}]}
        )
        assert args["path"] == "/tmp/a.py"

    def test_multi_file_codex_patch_reports_the_remainder(self):
        """The count must not be silently dropped."""
        args = canonical_tool_args(
            "apply_patch",
            {
                "changes": [
                    {"kind": "update", "path": "/tmp/a.py"},
                    {"kind": "update", "path": "/tmp/b.py"},
                    {"kind": "add", "path": "/tmp/c.py"},
                ]
            },
        )
        assert args["path"] == "/tmp/a.py +2 more"

    @pytest.mark.parametrize(
        "changes", [None, "not-a-list", [], [{"kind": "update"}], [42]]
    )
    def test_unusable_changes_yield_no_path(self, changes):
        args = canonical_tool_args("apply_patch", {"changes": changes})
        assert "path" not in args

    def test_canonicalize_returns_both_halves(self):
        name, args = canonicalize_tool_call("Bash", {"command": "ls"})
        assert (name, args) == ("terminal", {"command": "ls"})


class TestThreeLaneDisplayParity:
    """The same operation renders identically on every runtime."""

    @pytest.mark.parametrize(
        "lanes", [TERMINAL_LANES, READ_LANES, PATCH_LANES, TODO_LANES]
    )
    def test_verb_preview_and_emoji_match_across_lanes(self, lanes):
        rendered = [
            (
                get_tool_verb(name),
                tool_verb_connector(name),
                verb_drops_preview(name),
                build_tool_preview(name, args),
                get_tool_emoji(name),
                build_status_phrase(name, args),
            )
            for _label, name, args in lanes
        ]
        native = rendered[0]
        for (label, _name, _args), actual in zip(lanes[1:], rendered[1:]):
            assert actual == native, f"{label} lane diverged from native"

    def test_anchors_real_values_not_just_agreement(self):
        """Guards against the tables being emptied and parity passing vacuously."""
        assert get_tool_verb("Bash") == "Running"
        assert get_tool_verb("Read") == "Reading"
        assert get_tool_verb("exec_command") == "Running"
        assert "ls -la /tmp" in build_tool_preview("Bash", {"command": "ls -la /tmp"})

    def test_terminal_gate_sees_a_command_on_every_lane(self):
        """The gateway's fenced-code-block gate keys on canonical name + command.

        The gate itself lives inside ``TurnRunner`` and is not unit-testable
        here; this pins the inputs it branches on.
        """
        for label, name, args in TERMINAL_LANES:
            canon_name, canon_args = canonicalize_tool_call(name, args)
            assert canon_name == "terminal", label
            assert canon_args["command"].strip() == "ls -la /tmp", label

    def test_multi_file_patch_preview_survives_canonicalization(self):
        preview = build_tool_preview(
            "apply_patch",
            {
                "changes": [
                    {"kind": "update", "path": "/tmp/a.py"},
                    {"kind": "update", "path": "/tmp/b.py"},
                ]
            },
        )
        assert preview is not None
        assert "+1 more" in preview


class TestNonTerminalRuntimesUnaffected:
    """Canonicalization must not invent behaviour for unmapped tools."""

    def test_unmapped_tool_keeps_its_own_identity(self):
        """``ToolSearch`` has no native counterpart, so it stays itself.

        It later gained a *display verb*, which is a different decision: a
        verb describes what the user is watching, while identity decides
        which curated behaviour a tool inherits. Giving ToolSearch a verb
        must not give it another tool's identity — above all not
        ``terminal``, which would route it into the shell code-block path.
        """
        assert canonical_tool_name("ToolSearch") == "ToolSearch"
        assert canonical_tool_name("ToolSearch") != "terminal"

    def test_third_party_mcp_tool_renders_without_error(self):
        name = "mcp__linear__get_issue"
        assert get_tool_verb(name) is None
        assert canonical_tool_name(name) == name
        build_tool_preview(name, {"issue_id": "ENG-1"})

    def test_non_shell_tools_are_not_treated_as_terminal(self):
        for name in ("Read", "Edit", "WebSearch", "TodoWrite"):
            assert canonical_tool_name(name) != "terminal"
