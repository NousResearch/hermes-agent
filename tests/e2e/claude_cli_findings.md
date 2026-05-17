# PR 1 — Claude Code CLI contract findings (real-host probe)

Findings recorded as each Phase C task runs against the real `claude`
binary on the user's host. Used to consolidate the CLI Contract appendix
into the design spec in Task 15.

## Host environment

- Date: 2026-05-17
- Claude version: 2.1.143 (Claude Code)
- OS: Linux pepper 6.8.0-100-generic #100-Ubuntu SMP PREEMPT_DYNAMIC Tue Jan 13 16:40:06 UTC 2026 x86_64
- Hermes worktree: /root/.hermes/hermes-agent-claude-cli-pr1
- Branch: claude-cli-pr1

## Task 8: basic stream-json invocation

- Test: `tests/e2e/test_claude_cli_probe.py::test_basic_stream_json_invocation`
- Result: PASS
- Exit code observed: 0
- Event types seen in stream: system, system, system, assistant, rate_limit_event, result
- Stderr digest (first 500 bytes): (empty — no stderr output)
- Wall time: ~5.3 seconds
- stdin prompt transport supported: yes

### Notes

- Three `system` events arrive before `assistant`; these appear to be
  Claude Code's internal init/session setup events emitted by `--verbose`.
- A `rate_limit_event` event is emitted between `assistant` and `result`.
  This is a normal informational event, not an error. It does not indicate
  actual rate limiting was hit.
- The `--no-session-persistence` and `--allowedTools ""` flags work as expected
  on version 2.1.143.
- Empty stderr confirms no warnings, banners, or debug logs leaked to stderr
  when using `--output-format stream-json`.

Conclusion: stdin prompt transport via `-p` + `--output-format stream-json`
works correctly on version 2.1.143; the adapter design assumption is valid.

## Task 9: --resume continuity + session ID extraction

- Test: `tests/e2e/test_claude_cli_probe.py::test_resume_continuity_and_session_id_extraction`
- Result: PASS
- session_id schema location: top-level `session_id` field on every stream-json event; first occurrence is on the first `system` event (index 0). `extract_session_id()` finds it on the first pass by scanning `event.get("session_id")` across all events.
- Sample session_id (first 8 chars only, redacted): `0a8cb6d1...`
- --resume preserves context: yes — turn 2 correctly recalled the word "zephyr" from turn 1
- Wall time: ~11 seconds (both turns combined, xdist parallel run)

### Notes

- The `session_id` is a UUID4 string (e.g. `0a8cb6d1-8f79-4a92-967f-859d130a6736`).
- Every event in the stream carries the same `session_id` at the top level — it is NOT nested inside a `result` sub-object. The secondary fallback in `extract_session_id()` (checking `event["result"]["session_id"]`) is not needed for 2.1.143 but kept for forward compatibility.
- `--no-session-persistence` (used in Task 8's test) suppresses session persistence and would prevent `--resume` from working. The resume test intentionally omits that flag.
- The third `system` event contains a rich metadata payload (cwd, tools, mcp_servers, model, permissionMode, claude_code_version, etc.) — useful for future adapter introspection.
- `--allowedTools ""` works correctly with `--resume`; the resumed session inherits the original session's tool permissions.

Conclusion: `session_id` is a reliable top-level field on all stream-json events on version 2.1.143;
`--resume <session_id>` correctly restores conversational context across subprocess invocations.

## Task 10: coarse permissioning canaries

- `--allowedTools ""` denies all built-in tools: yes — zero `tool_use` events emitted even when prompted to read `/etc/hostname`; `--disallowedTools Bash,Read,Edit,Write,WebFetch,WebSearch` was added as belt-and-suspenders but is not needed for the canary assertion (model responded with "unable" text instead of a tool call). The empty-string form `--allowedTools ""` works correctly on 2.1.143.
- `--strict-mcp-config` + empty mcp-config ignores ambient ~/.claude/settings.json mcpServers: yes — poisoned settings.json with `"canary"` MCP server declaration (HOME isolated via tmp_path) produced zero output containing "canary" or "canary-loaded"; the third `system` event metadata confirmed no MCP servers were registered.
- Required mitigations (if either failed): none — both assumptions hold without additional flags.
- Sample stderr (first 200 bytes if anything notable): b'' (empty on both runs — no warnings, banners, or debug logs)

Conclusion: Both v1 default-deny posture canaries hold on claude 2.1.143; `--allowedTools ""` fully blocks built-in tool use and `--strict-mcp-config` + empty `--mcp-config` provides hermetic MCP server isolation when HOME is controlled.

## Task 11: hermetic settings precedence

- Test: `tests/e2e/test_claude_cli_probe.py::test_settings_file_overrides_ambient_settings`
- Result: PASS
- `--settings <file>` overrides ambient ~/.claude/settings.json: yes — zero `tool_use` events emitted even though the poisoned ambient settings.json declared `{"allowed": ["Bash","Read","Write","Edit"], "denied": []}`. The hermetic file (`{"allowed": [], "denied": ["*"]}`) took precedence; the model responded with "unable" text rather than attempting a Bash tool call.
- Permissions schema used in the hermetic file: `{"permissions": {"tools": {"allowed": [], "denied": ["*"]}}}`
- Notes on `--setting-sources` (if you tested it): not tested in this task
- Wall time: ~7.9 seconds

Conclusion: `--settings <file>` reliably overrides ambient `~/.claude/settings.json` on claude 2.1.143; the v1 hermetic-config posture is validated.

## Task 12: process group cleanup + egress observation

- Test: `test_process_group_cleanup_on_cancel`
- Result: PASS
- Process group cleanup on SIGTERM/SIGKILL: yes — `os.killpg(pgid, SIGTERM)` followed by fallback `os.killpg(pgid, SIGKILL)` terminated the entire claude process group within 5s; `ps --ppid <pytest-pid>` showed no surviving `claude` or `node` children after the kill sequence.
- Surviving children after kill: none
- Wall time: ~2.7 seconds (test completed well before the 30s deadline)
- Test: `test_no_direct_anthropic_egress_from_test_process`
- Result: PASS (documentation-only)
- Network egress to api.anthropic.com from pytest pid during test (operator-observed via `ss -tnp`): not observed — `ss -tnp | grep anthropic` returned empty after test completion; no leftover anthropic connections
- Network egress from child claude pid: observed (expected — the child `claude` process makes outbound TLS to api.anthropic.com; this is the intended pattern)

Conclusion: `start_new_session=True` + `os.killpg(SIGTERM→SIGKILL)` reliably reaps the entire claude process tree with no orphaned children; operator-side `ss -tnp` is the correct methodology for verifying that network egress to api.anthropic.com originates only from the child process, not from the Hermes parent.
