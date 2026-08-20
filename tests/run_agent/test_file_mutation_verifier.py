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
structurally impossible past the model while suppressing stale failures
when a later on-disk change proves recovery through another route.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading

import pytest
import run_agent as run_agent_module

from run_agent import (
    AIAgent,
    _FILE_MUTATING_TOOLS,
    _extract_error_preview,
    _extract_file_mutation_targets,
    _extract_landed_file_mutation_paths,
)


# ---------------------------------------------------------------------------
# _extract_file_mutation_targets
# ---------------------------------------------------------------------------


class TestExtractFileMutationTargets:
    def test_non_mutating_tool_returns_empty(self):
        assert _extract_file_mutation_targets("read_file", {"path": "/x"}) == []
        assert _extract_file_mutation_targets("terminal", {"command": "ls"}) == []



    def test_patch_replace_mode_returns_path(self):
        args = {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"}
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


    def test_patch_v4a_accepts_no_space_after_asterisks(self):
        """Match patch_parser / file_tools: ``***Update File:`` (no space)."""
        body = "***Update File: nospace.py\n"
        assert _extract_file_mutation_targets(
            "patch", {"mode": "patch", "patch": body}
        ) == ["nospace.py"]


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
    agent._turn_file_mutation_paths = set()
    agent._turn_file_mutation_fingerprint_bytes = 0
    agent._turn_file_mutation_fingerprint_budget_exhausted = False
    agent._turn_file_mutation_fingerprint_lock = threading.Lock()
    agent._turn_file_mutation_state_lock = threading.Lock()
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

    def test_disabled_verifier_skips_failure_fingerprinting(self, monkeypatch):
        agent = _bare_agent()
        monkeypatch.setattr(agent, "_file_mutation_verifier_enabled", lambda: False)
        monkeypatch.setattr(
            agent,
            "_snapshot_file_mutation_target",
            lambda *_args, **_kwargs: pytest.fail("disabled verifier fingerprinted a target"),
        )

        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"},
            json.dumps({"error": "not found"}),
            is_error=True,
        )

        assert agent._turn_failed_file_mutations == {}

    def test_disabled_verifier_skips_final_fingerprint(self, monkeypatch):
        agent = _bare_agent()
        agent._turn_failed_file_mutations["/tmp/a.md"] = {
            "tool": "patch",
            "error_preview": "not found",
            "resolved_path": "/tmp/a.md",
            "fingerprint": ("file", 1, "baseline"),
            "task_id": "default",
        }
        monkeypatch.setattr(agent, "_file_mutation_verifier_enabled", lambda: False)
        monkeypatch.setattr(
            agent,
            "_snapshot_file_mutation_target",
            lambda *_args, **_kwargs: pytest.fail("disabled verifier re-fingerprinted a target"),
        )

        assert agent._unresolved_file_mutation_failures() == {}

    def test_failure_uses_resolved_target_reported_by_tool(self, tmp_path, monkeypatch):
        agent = _bare_agent()
        actual = tmp_path / "actual.txt"
        wrong = tmp_path / "wrong.txt"
        actual.write_text("actual baseline\n", encoding="utf-8")
        wrong.write_text("wrong baseline\n", encoding="utf-8")
        monkeypatch.setattr(
            agent,
            "_resolve_file_mutation_target",
            lambda *_args, **_kwargs: str(wrong),
        )

        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "relative.txt", "old_string": "x", "new_string": "y"},
            json.dumps({
                "error": "Could not find old_string",
                "resolved_path": str(actual),
            }),
            is_error=True,
        )

        info = agent._turn_failed_file_mutations["relative.txt"]
        assert info["resolved_path"] == str(actual)
        assert info["fingerprint"] == (
            "file",
            actual.stat().st_size,
            hashlib.sha256(actual.read_bytes()).hexdigest(),
        )

    def test_patch_failure_reports_exact_resolved_target(self, tmp_path):
        from tools.file_tools import patch_tool

        target = tmp_path / "target.txt"
        target.write_text("actual content\n", encoding="utf-8")

        result = json.loads(patch_tool(
            mode="replace",
            path=str(target),
            old_string="missing content",
            new_string="replacement",
            task_id="mutation-verifier-test",
        ))

        assert result.get("error")
        assert result["resolved_path"] == str(target)
        assert result["resolved_paths"] == [str(target)]

    def test_v4a_failure_maps_exact_targets_when_optional_path_is_present(self, tmp_path):
        from tools.file_tools import patch_tool

        target = tmp_path / "target.txt"
        unrelated = tmp_path / "unrelated.txt"
        target.write_text("actual content\n", encoding="utf-8")
        unrelated.write_text("unrelated\n", encoding="utf-8")
        patch_body = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-missing content\n"
            "+replacement\n"
            "*** End Patch"
        )

        result = patch_tool(
            mode="patch",
            path=str(unrelated),
            patch=patch_body,
            task_id="mutation-verifier-test",
        )
        result_data = json.loads(result)
        assert result_data.get("error")
        assert result_data["resolved_path_map"][str(target)] == str(target)

        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "patch", "path": str(unrelated), "patch": patch_body},
            result,
            is_error=True,
        )

        assert list(agent._turn_failed_file_mutations) == [str(target)]
        assert agent._turn_failed_file_mutations[str(target)]["resolved_path"] == str(target)

    def test_stale_expected_state_cannot_mutate_next_turn(self):
        agent = _bare_agent()
        old_state = agent._turn_failed_file_mutations
        next_state = {
            "/tmp/a.md": {
                "tool": "patch",
                "error_preview": "next-turn failure",
            }
        }
        agent._turn_failed_file_mutations = next_state

        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"},
            json.dumps({"success": True}),
            is_error=False,
            expected_state=old_state,
        )

        assert agent._turn_failed_file_mutations is next_state
        assert next_state["/tmp/a.md"]["error_preview"] == "next-turn failure"

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
        assert agent._turn_file_mutation_paths == {"/tmp/a.md"}

    def test_success_clears_relative_failure_reported_as_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "a.md"
        target.write_text("before\n", encoding="utf-8")
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "a.md", "old_string": "x", "new_string": "y"},
            json.dumps({"error": "not found"}),
            is_error=True,
        )

        target.write_text("after\n", encoding="utf-8")
        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": str(target), "old_string": "before", "new_string": "after"},
            json.dumps({"success": True, "files_modified": [str(target)]}),
            is_error=False,
        )

        assert agent._turn_failed_file_mutations == {}

    @pytest.mark.parametrize(
        ("failed_path", "successful_path"),
        [
            ("src/a.py", "/workspace/src/a.py"),
            ("/workspace/src/a.py", "src/a.py"),
        ],
    )
    def test_remote_success_clears_lexically_equivalent_alias_only(
        self, monkeypatch, failed_path, successful_path,
    ):
        """Remote alias matching must not depend on host filesystem access."""
        import tools.file_tools as file_tools

        monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task_id: "docker")
        monkeypatch.setattr(
            file_tools,
            "_authoritative_workspace_root",
            lambda _task_id: "/workspace",
        )
        monkeypatch.setattr(
            run_agent_module,
            "Path",
            lambda _path: pytest.fail("remote verifier target touched the host filesystem"),
        )

        agent = _bare_agent()
        failed_patch = (
            f"*** Update File: {failed_path}\n"
            "*** Update File: src/sibling.py\n"
        )
        agent._record_file_mutation_result(
            "patch",
            {"mode": "patch", "patch": failed_patch},
            json.dumps({"error": "multi-file patch failed"}),
            is_error=True,
            task_id="remote-task",
        )

        assert agent._turn_failed_file_mutations[failed_path]["resolved_path"] == "/workspace/src/a.py"
        assert agent._turn_failed_file_mutations[failed_path]["fingerprint"] is None

        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": successful_path},
            json.dumps({"success": True, "files_modified": ["/workspace/src/a.py"]}),
            is_error=False,
            task_id="remote-task",
        )

        assert list(agent._turn_failed_file_mutations) == ["src/sibling.py"]

    def test_fingerprint_budget_is_aggregate_and_exhaustion_is_conservative(
        self, tmp_path, monkeypatch,
    ):
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        third = tmp_path / "third.txt"
        first.write_bytes(b"a" * 8)
        second.write_bytes(b"b" * 8)
        third.write_bytes(b"c" * 2)
        monkeypatch.setattr(run_agent_module, "_FILE_MUTATION_FINGERPRINT_TURN_MAX_BYTES", 12)
        monkeypatch.setattr(run_agent_module, "_FILE_MUTATION_FINGERPRINT_MAX_FILE_BYTES", 12)

        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {
                "mode": "patch",
                "patch": (
                    f"*** Update File: {first}\n"
                    f"*** Update File: {second}\n"
                    f"*** Update File: {third}\n"
                ),
            },
            json.dumps({"error": "multi-file patch failed"}),
            is_error=True,
        )

        state = agent._turn_failed_file_mutations
        assert state[str(first)]["fingerprint"] is not None
        assert state[str(second)]["fingerprint"] is None
        assert state[str(third)]["fingerprint"] is None
        assert agent._turn_file_mutation_fingerprint_bytes == 8

        first.write_bytes(b"changed!")
        second.write_bytes(b"changed!")
        third.write_bytes(b"changed!")
        unresolved = agent._unresolved_file_mutation_failures()

        assert list(unresolved) == [str(first), str(second), str(third)]
        assert agent._turn_file_mutation_fingerprint_bytes <= 12
        footer = agent._format_file_mutation_failure_footer(unresolved)
        assert "3 failed file mutation target(s) remain unresolved" in footer


    def test_landed_paths_prefer_resolved_tool_result(self):
        paths = _extract_landed_file_mutation_paths(
            "patch",
            {"mode": "replace", "path": "src/app.py"},
            json.dumps({
                "success": True,
                "files_modified": ["/tmp/project/src/app.py"],
            }),
        )

        assert paths == ["/tmp/project/src/app.py"]

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

    def test_terminal_cli_recovery_is_detected_from_disk(self, tmp_path):
        target = tmp_path / "config.yaml"
        target.write_text("enabled: false\n", encoding="utf-8")
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": str(target)},
            json.dumps({"error": "protected config; use hermes config set"}),
            is_error=True,
        )

        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(target)!r}).write_text('enabled: true\\n', encoding='utf-8')"
                ),
            ],
            check=True,
        )

        assert agent._unresolved_file_mutation_failures() == {}

    def test_created_target_recovers_failed_write_file(self, tmp_path):
        target = tmp_path / "new.txt"
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "write_file",
            {"path": str(target), "content": "created later"},
            json.dumps({"error": "write refused"}),
            is_error=True,
        )

        target.write_text("created later", encoding="utf-8")

        assert agent._unresolved_file_mutation_failures() == {}

    def test_unchanged_target_remains_unresolved(self, tmp_path):
        target = tmp_path / "unchanged.txt"
        target.write_text("before\n", encoding="utf-8")
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": str(target)},
            json.dumps({"error": "old_string not found"}),
            is_error=True,
        )

        assert set(agent._unresolved_file_mutation_failures()) == {str(target)}

    def test_deleted_target_remains_unresolved(self, tmp_path):
        target = tmp_path / "deleted.txt"
        target.write_text("before\n", encoding="utf-8")
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": str(target)},
            json.dumps({"error": "old_string not found"}),
            is_error=True,
        )

        target.unlink()

        assert set(agent._unresolved_file_mutation_failures()) == {str(target)}

    def test_v4a_recovery_filters_only_the_changed_sibling(self, tmp_path):
        changed = tmp_path / "changed.txt"
        unchanged = tmp_path / "unchanged.txt"
        changed.write_text("before\n", encoding="utf-8")
        unchanged.write_text("before\n", encoding="utf-8")
        patch_body = (
            f"*** Update File: {changed}\n"
            f"*** Update File: {unchanged}\n"
        )
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "patch", "patch": patch_body},
            json.dumps({"error": "multi-file patch failed"}),
            is_error=True,
        )

        changed.write_text("after\n", encoding="utf-8")

        assert set(agent._unresolved_file_mutation_failures()) == {str(unchanged)}

    def test_later_failed_attempt_refreshes_recovery_baseline(self, tmp_path):
        target = tmp_path / "retry.txt"
        target.write_text("before\n", encoding="utf-8")
        agent = _bare_agent()
        failed = json.dumps({"error": "old_string not found"})
        args = {"mode": "replace", "path": str(target)}

        agent._record_file_mutation_result("patch", args, failed, is_error=True)
        target.write_text("changed between attempts\n", encoding="utf-8")
        agent._record_file_mutation_result("patch", args, failed, is_error=True)

        assert set(agent._unresolved_file_mutation_failures()) == {str(target)}





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
        assert "1 failed file mutation target(s) remain unresolved" in out
        assert "No later change to these targets was detected" in out
        assert "were NOT modified" not in out
        assert "/tmp/a.md" in out
        assert "Could not find old_string" in out
        assert "git status" in out  # user-actionable hint

    def test_truncation_at_10_entries(self):
        failed = {
            f"/tmp/f{i}.md": {"tool": "patch", "error_preview": "err"}
            for i in range(15)
        }
        out = AIAgent._format_file_mutation_failure_footer(failed)
        assert "15 failed file mutation target(s) remain unresolved" in out
        assert "… and 5 more" in out
        # Ten file bullets + header + "and X more" line
        lines = out.split("\n")
        bullet_lines = [ln for ln in lines if ln.lstrip().startswith("•")]
        assert len(bullet_lines) == 11  # 10 shown + 1 summary


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

    def test_config_read_once_then_cached(self, monkeypatch):
        """Measured-work pin: the config lookup happens once per agent.

        The footer gate runs at the end of every turn, so a fresh
        ``load_config()`` per call is wasted work (measured ~0.9 ms/call on
        a warm mtime-cache on this host; the sibling per-turn-config kill in
        #74211 removed exactly this class of read).  The config read must be
        cached after the first call; the env-var override must still win on
        every call, cached or not.
        """
        monkeypatch.delenv("HERMES_FILE_MUTATION_VERIFIER", raising=False)
        agent = _bare_agent()
        calls = {"n": 0}

        import hermes_cli.config as _cfg_mod

        def counting_load():
            calls["n"] += 1
            return {"display": {"file_mutation_verifier": True}}

        monkeypatch.setattr(_cfg_mod, "load_config", counting_load)

        # First call reads config and caches the result.
        assert agent._file_mutation_verifier_enabled() is True
        assert calls["n"] == 1
        # Subsequent calls must not re-read config.
        assert agent._file_mutation_verifier_enabled() is True
        assert agent._file_mutation_verifier_enabled() is True
        assert calls["n"] == 1
        # Env override stays authoritative even after the cache is warm.
        monkeypatch.setenv("HERMES_FILE_MUTATION_VERIFIER", "0")
        assert agent._file_mutation_verifier_enabled() is False
        assert calls["n"] == 1  # env path never touches config

    def test_cache_respects_config_value(self, monkeypatch):
        """A disabled config value is cached as False, not re-read."""
        monkeypatch.delenv("HERMES_FILE_MUTATION_VERIFIER", raising=False)
        agent = _bare_agent()

        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(
            _cfg_mod, "load_config", lambda: {"display": {"file_mutation_verifier": False}}
        )
        assert agent._file_mutation_verifier_enabled() is False
        # Warm cache: flip the underlying config; the agent still reports the
        # cached value (same next-session semantics as _credits_notices_enabled).
        monkeypatch.setattr(
            _cfg_mod, "load_config", lambda: {"display": {"file_mutation_verifier": True}}
        )
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
