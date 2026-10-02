"""A templated compaction stub must carry the outcome of the call (#131244).

`_sum_template` formats from the tool arguments plus the result's length, so every tool routed
through it dropped the outcome: a refused `read_file`, a rate-limited `web_search`, a rejected
`cronjob_manage`, an unknown `process_manage` session and a `text_to_speech` call with no voice all
compressed into the same stub as the success they never were. The post-compaction agent then
reported the success.

Complementary to #131247, which covers the two mutation summarizers with their own call paths
(`write_file`, `patch`); this covers the templated family and does not touch those two.

The no-op-on-success half is pinned too — the suffix must stay absent when a payload reports no
failure, or every healthy stub grows noise.
"""

import json

from agent.context_compressor import _summarize_tool_result
from tools.file_tools import patch_tool
from tools.file_tools_read_tracking import reset_file_dedup


def _stub(tool_name, args, result):
    return _summarize_tool_result(tool_name, json.dumps(args), result)


def _failure(message, **extra):
    return json.dumps({"error": message, **extra})


# tool_name -> (args, a payload that reports failure, the reason the stub must carry)
FAILING_CALLS = [
    ("read_file", {"path": "gone.py", "offset": 1}, "File not found: gone.py"),
    ("web_search", {"query": "hermes agent"}, "rate limited"),
    ("memory", {"action": "add", "target": "a note"}, "unknown action"),
    ("cronjob_manage", {"action": "create"}, "a job with that name already exists"),
    ("process_manage", {"action": "poll", "session_id": "s1"}, "no such session"),
    ("text_to_speech", {}, "no voice available"),
]

# The same tools with a payload that reports no failure: the stub must be unchanged.
HEALTHY_PAYLOADS = {
    "read_file": json.dumps({"content": "x", "total_lines": 1}),
    "web_search": json.dumps({"results": [{"title": "hit"}]}),
    "memory": json.dumps({"success": True, "stored": "a note"}),
    "cronjob_manage": json.dumps({"success": True, "job": "nightly"}),
    "process_manage": json.dumps({"exit_code": 0}),
    "text_to_speech": json.dumps({"success": True, "path": "/tmp/out.wav"}),
}


class TestTemplatedStubCarriesOutcome:
    def test_each_failing_call_is_marked_failed(self):
        for tool_name, args, reason in FAILING_CALLS:
            stub = _stub(tool_name, args, _failure(reason))
            assert "FAILED: " + reason in stub, f"{tool_name}: {stub}"
            assert "\n" not in stub, f"{tool_name}: {stub}"

    def test_a_failure_without_a_message_is_still_marked(self):
        stub = _stub("cronjob_manage", {"action": "create"}, json.dumps({"success": False}))

        assert "FAILED" in stub, stub

    def test_successful_calls_gain_no_suffix(self):
        for tool_name, args, _reason in FAILING_CALLS:
            stub = _stub(tool_name, args, HEALTHY_PAYLOADS[tool_name])
            assert "FAILED" not in stub, f"{tool_name}: {stub}"
            # The pre-existing wording is untouched — this test passes identically before the fix.
            assert stub.startswith("[") and stub == stub.strip() and "\n" not in stub, f"{tool_name}: {stub}"

    def test_failed_read_keeps_its_place_and_offset(self):
        stub = _stub("read_file", {"path": "gone.py", "offset": 40}, _failure("File not found: gone.py"))

        assert stub.startswith("[read_file] read gone.py from line 40"), stub

    def test_rejected_patch_is_marked(self, tmp_path):
        target = tmp_path / "code.py"
        target.write_text("alpha\nbeta\n")
        reset_file_dedup()

        rejected = patch_tool("replace", path=str(target), old_string="gamma", new_string="delta", task_id="t-tmpl")
        assert "error" in json.loads(rejected), rejected

        stub = _stub("patch", {"mode": "replace", "path": str(target)}, rejected)

        assert "FAILED: " in stub, stub
        assert stub.startswith(f"[patch] replace in {target}"), stub


class TestSkillStubsKeepTheirOutcome:
    def test_failed_skill_manage_is_marked(self):
        stub = _stub("skill_manage", {"action": "update", "name": "absent"}, _failure("not found", success=False))

        assert "FAILED: not found" in stub, stub

    def test_successful_skill_manage_is_not_marked(self):
        stub = _stub("skill_manage", {"action": "update", "name": "absent"}, json.dumps({"success": True}))

        assert "FAILED" not in stub, stub


def test_summarizer_never_raises_on_a_hostile_payload():
    """The outer guard must keep holding: a summary may never crash compression."""
    for tool_name, args, _reason in FAILING_CALLS:
        for hostile in ("", "null", "[1, 2, 3]", '{"error": {"nested": true}}', "not json at all"):
            assert isinstance(_stub(tool_name, args, hostile), str)


class TestFailuresTheTopLevelProbeCannotSee:
    """Two real shapes carry their outcome where a top-level ``error`` / ``success is False`` cannot.

    A ``cronjob_manage(action="run")`` whose job died returns ``{"success": True, "job": {...
    "execution_error": ...}}`` — the tool call itself worked, the job did not. A
    ``process_manage(action="poll")`` on a process that exited non-zero reports ``exit_code`` and
    ``completion_reason`` and nothing else. Both compressed into the byte-identical success stub.
    """

    def test_a_failed_cron_run_is_marked(self):
        payload = json.dumps({"success": True, "job": {"name": "nightly", "execution_success": False,
                                                       "execution_error": "Traceback: agent exited with code 1"}})

        stub = _stub("cronjob_manage", {"action": "run"}, payload)

        assert "FAILED: Traceback: agent exited with code 1" in stub, stub
        assert stub.startswith("[cronjob] run"), stub

    def test_a_cron_run_lost_to_the_scheduler_is_marked(self):
        payload = json.dumps({"success": True, "job": {"execution_skipped":
                             "Already being fired by the scheduler; not run again."}})

        stub = _stub("cronjob_manage", {"action": "run"}, payload)

        assert "FAILED: Already being fired by the scheduler" in stub, stub

    def test_a_failed_run_without_a_message_is_still_marked(self):
        payload = json.dumps({"success": True, "job": {"execution_success": False}})

        assert "FAILED" in _stub("cronjob_manage", {"action": "run"}, payload)

    def test_a_stored_error_from_an_earlier_run_is_not_marked(self):
        """``job`` carries the job's own state, so only this call's outcome may mark the stub."""
        payload = json.dumps({"success": True, "job": {"name": "nightly", "error": "last run failed",
                                                       "execution_success": True}})

        stub = _stub("cronjob_manage", {"action": "poll"}, payload)

        assert stub == "[cronjob] poll", stub

    def test_a_non_zero_exit_is_marked(self):
        payload = json.dumps({"session_id": "proc_1", "status": "exited", "exit_code": 1,
                              "completion_reason": "nonzero_exit", "termination_source": "child"})

        stub = _stub("process_manage", {"action": "poll", "session_id": "proc_1"}, payload)

        assert "FAILED" in stub and "exit code 1" in stub, stub
        assert stub.startswith("[process] poll session=proc_1"), stub

    def test_a_clean_exit_is_not_marked(self):
        payload = json.dumps({"session_id": "proc_1", "status": "exited", "exit_code": 0})

        assert _stub("process_manage", {"action": "poll", "session_id": "proc_1"}, payload) == \
            "[process] poll session=proc_1"

    def test_a_still_running_poll_is_not_marked(self):
        payload = json.dumps({"session_id": "proc_1", "status": "running", "pid": 4242})

        assert _stub("process_manage", {"action": "poll", "session_id": "proc_1"}, payload) == \
            "[process] poll session=proc_1"
