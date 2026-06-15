"""Tests for the per-turn file-mutation verifier footer.

Covers the three moving pieces:

1. ``_extract_file_mutation_targets`` — pulls file paths from write_file /
   patch (replace + V4A) tool-call argument dicts.
2. ``AIAgent._record_file_mutation_result`` — builds the per-turn state
   dict, removing entries when a later success supersedes an earlier
   failure for the same path.
3. ``AIAgent._format_file_mutation_failure_footer`` — renders the dict
   as a user-visible advisory.

Regression target: the "Ben Eng llm-wiki" session where grok-4.1-fast
batched parallel patches, half failed, and the model summarised the
turn claiming every file was edited.  This verifier makes over-claiming
structurally impossible past the model: the user always sees the real
list of files that did NOT change.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import (
    AIAgent,
    _FILE_MUTATING_TOOLS,
    _extract_error_preview,
    _extract_file_mutation_targets,
)
from agent.conversation_loop import (
    MAX_FILE_MUTATION_REMEDIATION_RETRIES,
    _file_mutation_remediation_prompt,
    _should_remediate_file_mutation_failures,
)


def _make_tool_defs(*names: str) -> list[dict]:
    """Build minimal tool definitions accepted by ``AIAgent.__init__``."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name: str, arguments: str, call_id: str = "call_patch"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content: str = "", *, tool_calls=None, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            ),
        ],
        model="test/model",
        usage=None,
    )


# ---------------------------------------------------------------------------
# _extract_file_mutation_targets
# ---------------------------------------------------------------------------


class TestExtractFileMutationTargets:
    def test_non_mutating_tool_returns_empty(self):
        assert _extract_file_mutation_targets("read_file", {"path": "/x"}) == []
        assert _extract_file_mutation_targets("terminal", {"command": "ls"}) == []

    def test_write_file_returns_single_path(self):
        out = _extract_file_mutation_targets("write_file", {"path": "/tmp/a.md", "content": "x"})
        assert out == ["/tmp/a.md"]

    def test_write_file_missing_path_returns_empty(self):
        assert _extract_file_mutation_targets("write_file", {"content": "x"}) == []

    def test_patch_replace_mode_returns_path(self):
        args = {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"}
        assert _extract_file_mutation_targets("patch", args) == ["/tmp/a.md"]

    def test_patch_default_mode_is_replace(self):
        # Mode omitted — schema default is ``replace``.
        args = {"path": "/tmp/a.md", "old_string": "x", "new_string": "y"}
        assert _extract_file_mutation_targets("patch", args) == ["/tmp/a.md"]

    def test_patch_v4a_single_file(self):
        body = (
            "*** Begin Patch\n"
            "*** Update File: /tmp/a.md\n"
            "@@ ctx @@\n"
            " line1\n"
            "-bad\n"
            "+good\n"
            "*** End Patch\n"
        )
        args = {"mode": "patch", "patch": body}
        assert _extract_file_mutation_targets("patch", args) == ["/tmp/a.md"]

    def test_patch_v4a_multi_file(self):
        body = (
            "*** Begin Patch\n"
            "*** Update File: /tmp/a.md\n"
            "@@ @@\n-a\n+b\n"
            "*** Add File: /tmp/new.md\n"
            "+fresh\n"
            "*** Delete File: /tmp/old.md\n"
            "*** End Patch\n"
        )
        args = {"mode": "patch", "patch": body}
        paths = _extract_file_mutation_targets("patch", args)
        assert paths == ["/tmp/a.md", "/tmp/new.md", "/tmp/old.md"]

    def test_patch_v4a_missing_body_returns_empty(self):
        assert _extract_file_mutation_targets("patch", {"mode": "patch"}) == []
        assert _extract_file_mutation_targets("patch", {"mode": "patch", "patch": ""}) == []


# ---------------------------------------------------------------------------
# _extract_error_preview
# ---------------------------------------------------------------------------


class TestExtractErrorPreview:
    def test_json_error_field_preferred(self):
        raw = json.dumps({"success": False, "error": "Could not find old_string in /tmp/x"})
        assert _extract_error_preview(raw) == "Could not find old_string in /tmp/x"

    def test_plain_string_falls_through(self):
        assert _extract_error_preview("Error executing tool: boom") == "Error executing tool: boom"

    def test_long_preview_truncated(self):
        long = "x" * 500
        out = _extract_error_preview(long, max_len=50)
        assert len(out) <= 50
        assert out.endswith("…")

    def test_none_returns_empty(self):
        assert _extract_error_preview(None) == ""


# ---------------------------------------------------------------------------
# _record_file_mutation_result — state transitions
# ---------------------------------------------------------------------------


def _bare_agent() -> AIAgent:
    """Skip __init__ and only attach the per-turn state dict.

    AIAgent.__init__ takes ~60 parameters and touches network, auth, and
    the filesystem.  For these tests we only need the two methods —
    ``_record_file_mutation_result`` and ``_format_file_mutation_failure_footer``.
    Using ``object.__new__`` mirrors the gateway-test pattern documented in
    the agent pitfalls list.
    """
    agent = object.__new__(AIAgent)
    agent._turn_failed_file_mutations = {}
    return agent


class TestRecordFileMutationResult:
    def test_non_mutating_tool_ignored(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "read_file", {"path": "/tmp/x"}, "{}", is_error=True,
        )
        assert agent._turn_failed_file_mutations == {}

    def test_failure_recorded(self):
        agent = _bare_agent()
        result = json.dumps({"success": False, "error": "Could not find old_string"})
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"},
            result, is_error=True,
        )
        state = agent._turn_failed_file_mutations
        assert "/tmp/a.md" in state
        assert state["/tmp/a.md"]["tool"] == "patch"
        assert "Could not find old_string" in state["/tmp/a.md"]["error_preview"]

    def test_success_removes_prior_failure(self):
        agent = _bare_agent()
        # First attempt fails
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"},
            json.dumps({"error": "not found"}), is_error=True,
        )
        assert "/tmp/a.md" in agent._turn_failed_file_mutations
        # Second attempt with corrected old_string succeeds
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "real", "new_string": "fixed"},
            json.dumps({"success": True, "diff": "..."}), is_error=False,
        )
        assert agent._turn_failed_file_mutations == {}

    def test_write_file_with_lint_error_counts_as_landed(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "write_file",
            {"path": "/tmp/a.py", "content": "bad"},
            json.dumps({"error": "write failed"}),
            is_error=True,
        )
        assert "/tmp/a.py" in agent._turn_failed_file_mutations

        result = json.dumps({
            "bytes_written": 24,
            "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
        })

        agent._record_file_mutation_result(
            "write_file",
            {"path": "/tmp/a.py", "content": "def nope(:\n"},
            result,
            is_error=True,
        )

        assert agent._turn_failed_file_mutations == {}

    def test_patch_with_lsp_diagnostics_counts_as_landed(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "/tmp/a.py", "old_string": "x", "new_string": "y"},
            json.dumps({"error": "Could not find old_string"}),
            is_error=True,
        )
        assert "/tmp/a.py" in agent._turn_failed_file_mutations

        result = json.dumps({
            "success": True,
            "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
            "files_modified": ["/tmp/a.py"],
            "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
        })

        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "/tmp/a.py", "old_string": "x", "new_string": "y"},
            result,
            is_error=True,
        )

        assert agent._turn_failed_file_mutations == {}

    def test_repeated_failure_keeps_first_error(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "v1", "new_string": "y"},
            json.dumps({"error": "first error"}), is_error=True,
        )
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "v2", "new_string": "y"},
            json.dumps({"error": "second error"}), is_error=True,
        )
        # Keep the original error — swapping to the latest would obscure
        # the initial root cause.
        assert "first error" in agent._turn_failed_file_mutations["/tmp/a.md"]["error_preview"]

    def test_v4a_multi_file_all_tracked(self):
        agent = _bare_agent()
        body = (
            "*** Begin Patch\n"
            "*** Update File: /tmp/a.md\n@@ @@\n-a\n+b\n"
            "*** Update File: /tmp/b.md\n@@ @@\n-a\n+b\n"
            "*** End Patch\n"
        )
        agent._record_file_mutation_result(
            "patch", {"mode": "patch", "patch": body},
            json.dumps({"error": "parse failure"}), is_error=True,
        )
        assert set(agent._turn_failed_file_mutations) == {"/tmp/a.md", "/tmp/b.md"}

    def test_no_state_dict_silent_noop(self):
        """When called outside run_conversation the state dict is absent.

        The record helper must never raise — a tool dispatched from, say,
        a direct ``chat()`` call should not blow up the call site just
        because the verifier state hasn't been initialised.
        """
        agent = object.__new__(AIAgent)  # no state attached
        # Should not raise
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md"},
            json.dumps({"error": "x"}), is_error=True,
        )

    def test_missing_path_arg_recorded_nowhere(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch", {"mode": "replace"},  # no path
            json.dumps({"error": "path required"}), is_error=True,
        )
        # No path → nothing to key on, state stays empty.  The per-turn
        # state is about file paths, not individual tool-call IDs.
        assert agent._turn_failed_file_mutations == {}


# ---------------------------------------------------------------------------
# _format_file_mutation_failure_footer
# ---------------------------------------------------------------------------


class TestFormatFooter:
    def test_empty_returns_empty_string(self):
        assert AIAgent._format_file_mutation_failure_footer({}) == ""

    def test_single_failure(self):
        out = AIAgent._format_file_mutation_failure_footer(
            {"/tmp/a.md": {"tool": "patch", "error_preview": "Could not find old_string"}},
        )
        assert "1 file(s) were NOT modified" in out
        assert "/tmp/a.md" in out
        assert "Could not find old_string" in out
        assert "git status" in out  # user-actionable hint

    def test_truncation_at_10_entries(self):
        failed = {
            f"/tmp/f{i}.md": {"tool": "patch", "error_preview": "err"}
            for i in range(15)
        }
        out = AIAgent._format_file_mutation_failure_footer(failed)
        assert "15 file(s) were NOT modified" in out
        assert "… and 5 more" in out
        # Ten file bullets + header + "and X more" line
        lines = out.split("\n")
        bullet_lines = [ln for ln in lines if ln.lstrip().startswith("•")]
        assert len(bullet_lines) == 11  # 10 shown + 1 summary

    def test_paths_are_backtick_wrapped(self):
        """Footer paths must be inline-code wrapped so the gateway's bare-path
        media extractor can't auto-attach them (#35584 defense-in-depth)."""
        out = AIAgent._format_file_mutation_failure_footer(
            {"/home/u/.hermes/config.yaml": {
                "tool": "patch",
                "error_preview": (
                    "Write denied: '/home/u/.hermes/config.yaml' is a "
                    "protected system/credential file."
                ),
            }},
        )
        # Path still human-readable.
        assert "/home/u/.hermes/config.yaml" in out
        # Bullet path is backticked.
        assert "`/home/u/.hermes/config.yaml`" in out
        # The path echoed inside the preview is ALSO backticked (the real
        # file_operations.py denial message embeds it in single quotes, which
        # do NOT block the gateway extractor's regex).
        assert "'`/home/u/.hermes/config.yaml`'" in out
        # No double-backticking anywhere.
        assert "``" not in out

    def test_footer_path_not_extracted_by_gateway(self):
        """End-to-end: the gateway's extract_local_files must NOT pull a
        config.yaml path out of the rendered footer (#35584)."""
        import os
        import tempfile
        from gateway.platforms.base import BasePlatformAdapter

        tmp = tempfile.mkdtemp(prefix="hermes_footer_")
        try:
            cfg = os.path.join(tmp, "config.yaml")
            with open(cfg, "w") as fh:
                fh.write("openrouter_api_key: sk-LEAK\n")
            footer = AIAgent._format_file_mutation_failure_footer(
                {cfg: {
                    "tool": "patch",
                    "error_preview": (
                        f"Write denied: '{cfg}' is a protected "
                        "system/credential file."
                    ),
                }},
            )
            response = "I updated your config.\n\n" + footer
            paths, _ = BasePlatformAdapter.extract_local_files(response)
            assert paths == [], f"footer leaked deliverable path(s): {paths}"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)



# ---------------------------------------------------------------------------
# automatic remediation nudge before the final verifier footer
# ---------------------------------------------------------------------------


class TestFileMutationAutoRemediation:
    def _agent(self, *, failed=True, retries=0, enabled=True, budget_remaining=3, max_iterations=8):
        def _enabled():
            return enabled

        return SimpleNamespace(
            _turn_failed_file_mutations=(
                {"/tmp/a.md": {"tool": "patch", "error_preview": "Could not find old_string"}}
                if failed else {}
            ),
            _turn_file_mutation_remediation_retries=retries,
            _file_mutation_verifier_enabled=_enabled,
            max_iterations=max_iterations,
            iteration_budget=SimpleNamespace(remaining=budget_remaining),
            _budget_grace_call=False,
        )

    def test_failed_edit_requests_one_automatic_remediation_turn(self):
        agent = self._agent()
        assert _should_remediate_file_mutation_failures(agent, api_call_count=2) is True

    def test_no_remediation_when_no_failed_edits(self):
        agent = self._agent(failed=False)
        assert _should_remediate_file_mutation_failures(agent, api_call_count=2) is False

    def test_remediation_is_bounded_then_footer_can_surface(self):
        agent = self._agent(retries=MAX_FILE_MUTATION_REMEDIATION_RETRIES)
        assert _should_remediate_file_mutation_failures(agent, api_call_count=2) is False

    def test_disabled_verifier_disables_auto_remediation_too(self):
        agent = self._agent(enabled=False)
        assert _should_remediate_file_mutation_failures(agent, api_call_count=2) is False

    def test_no_remediation_without_iteration_headroom(self):
        agent = self._agent(budget_remaining=0)
        assert _should_remediate_file_mutation_failures(agent, api_call_count=2) is False

    def test_no_remediation_at_max_iterations(self):
        agent = self._agent(max_iterations=3)
        assert _should_remediate_file_mutation_failures(agent, api_call_count=3) is False

    def test_remediation_prompt_tells_model_to_repair_not_report_done(self):
        failed = {
            "/tmp/a.md": {
                "tool": "patch",
                "error_preview": "Could not find old_string",
            },
        }
        prompt = _file_mutation_remediation_prompt(failed)
        assert "follow-up required before final answer" in prompt
        assert "repair the failed edit" in prompt
        assert "Do not claim those edits succeeded" in prompt
        assert "/tmp/a.md" in prompt
        assert "Could not find old_string" in prompt

    def test_failed_patch_gets_followup_turn_before_footer_only_final(self, monkeypatch):
        """A failed patch must not be reported only as a final-answer footer.

        The loop should give the model one synthetic continuation turn with
        the unresolved failed edit evidence before falling back to the footer.
        """
        import agent.redact as redact_mod

        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False)
        dummy_api_key = "test-" "key-1234567890"
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("patch")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key=dummy_api_key,
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        patch_args = {
            "mode": "replace",
            "path": "/tmp/a.md",
            "old_string": "missing",
            "new_string": "fixed",
        }
        responses = [
            _mock_response(
                tool_calls=[
                    _mock_tool_call(
                        "patch",
                        json.dumps(patch_args),
                    ),
                ],
                finish_reason="tool_calls",
            ),
            _mock_response(content="I updated `/tmp/a.md` with API_KEY=dummysecretvalue12345."),
            _mock_response(content="I could not repair `/tmp/a.md` after inspection."),
        ]
        api_messages: list[list[dict]] = []

        def _fake_api_call(api_kwargs):
            api_messages.append(list(api_kwargs.get("messages", [])))
            return responses.pop(0)

        agent.client = MagicMock()
        setattr(agent, "_interruptible_api_call", _fake_api_call)
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None

        failed_patch_result = json.dumps({
            "success": False,
            "error": "Could not find old_string",
        })
        with patch("run_agent.handle_function_call", return_value=failed_patch_result):
            result = agent.run_conversation("Patch /tmp/a.md")

        assert result["completed"] is True
        assert result["api_calls"] == 3
        assert "could not repair" in result["final_response"]
        assert "1 file(s) were NOT modified" in result["final_response"]

        followup_request = api_messages[2]
        followup_payload = json.dumps(followup_request)
        assert "API_KEY=dummysecretvalue12345" not in followup_payload
        assert "_file_mutation_remediation_synthetic" not in followup_payload
        synthetic_followups = [
            msg for msg in followup_request
            if "follow-up required before final answer" in str(msg.get("content", ""))
        ]
        assert synthetic_followups
        assert "follow-up required before final answer" in synthetic_followups[-1]["content"]
        assert "Could not find old_string" in synthetic_followups[-1]["content"]
        assert getattr(agent, "_turn_file_mutation_remediation_retries") == 1

    def test_successful_remediation_clears_failure_and_suppresses_footer(self):
        """A successful remediation edit should clear verifier state.

        This proves the automatic follow-up path does not leave a stale
        "NOT modified" footer after the model repairs the failed edit.
        """
        dummy_api_key = "test-" "key-1234567890"
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("patch")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key=dummy_api_key,
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        patch_args = {
            "mode": "replace",
            "path": "/tmp/a.md",
            "old_string": "missing",
            "new_string": "fixed",
        }
        repair_args = {
            "mode": "replace",
            "path": "/tmp/a.md",
            "old_string": "actual",
            "new_string": "fixed",
        }
        responses = [
            _mock_response(
                tool_calls=[_mock_tool_call("patch", json.dumps(patch_args), "call_fail")],
                finish_reason="tool_calls",
            ),
            _mock_response(content="I updated `/tmp/a.md` successfully."),
            _mock_response(
                tool_calls=[_mock_tool_call("patch", json.dumps(repair_args), "call_repair")],
                finish_reason="tool_calls",
            ),
            _mock_response(content="I repaired `/tmp/a.md` successfully."),
        ]

        def _fake_api_call(api_kwargs):
            return responses.pop(0)

        agent.client = MagicMock()
        setattr(agent, "_interruptible_api_call", _fake_api_call)
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None

        tool_results = [
            json.dumps({"success": False, "error": "Could not find old_string"}),
            json.dumps({"success": True, "diff": "--- a/tmp/a.md\n+++ b/tmp/a.md\n"}),
        ]
        with patch("run_agent.handle_function_call", side_effect=tool_results):
            result = agent.run_conversation("Patch /tmp/a.md")

        assert result["completed"] is True
        assert result["api_calls"] == 4
        assert "I repaired" in result["final_response"]
        assert "were NOT modified" not in result["final_response"]
        assert getattr(agent, "_turn_failed_file_mutations") == {}
        assert getattr(agent, "_turn_file_mutation_remediation_retries") == 1


# ---------------------------------------------------------------------------
# _file_mutation_verifier_enabled — env + config precedence
# ---------------------------------------------------------------------------


class TestVerifierEnabled:
    def test_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_FILE_MUTATION_VERIFIER", raising=False)
        agent = _bare_agent()
        # With no env and no config present, safe default is True.
        # load_config may surface a user config.yaml in some envs — stub it.
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(_cfg_mod, "load_config", lambda: {})
        assert agent._file_mutation_verifier_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
    def test_env_disables(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_FILE_MUTATION_VERIFIER", value)
        agent = _bare_agent()
        assert agent._file_mutation_verifier_enabled() is False

    def test_env_enables_over_config(self, monkeypatch):
        monkeypatch.setenv("HERMES_FILE_MUTATION_VERIFIER", "1")
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(
            _cfg_mod, "load_config",
            lambda: {"display": {"file_mutation_verifier": False}},
        )
        agent = _bare_agent()
        assert agent._file_mutation_verifier_enabled() is True

    def test_config_disables_when_no_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_FILE_MUTATION_VERIFIER", raising=False)
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(
            _cfg_mod, "load_config",
            lambda: {"display": {"file_mutation_verifier": False}},
        )
        agent = _bare_agent()
        assert agent._file_mutation_verifier_enabled() is False


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_file_mutating_tools_set_shape():
    """write_file + patch are the only tools the verifier tracks.

    Guard rail: if someone adds a third file-mutating tool (e.g. a new
    ``append_file``), they should also audit whether the verifier should
    track it.  This test fails loudly on unilateral additions.
    """
    assert _FILE_MUTATING_TOOLS == frozenset({"write_file", "patch"})
