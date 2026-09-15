# RFC: Hermes context in the Codex app-server runtime

**Status:** proposed; maintainer decisions requested before implementation.

**Scope:** context handoff and ownership in the existing optional runtime.

**Coordination:** [issue #74712](https://github.com/NousResearch/hermes-agent/issues/74712#issuecomment-5617443518).

Hermes can prepare a profile's instructions and conversation context without
delivering them to the Codex app-server turn. This RFC proposes the contract for
closing that gap while retaining Codex's native execution loop and history. It
provides a reviewable basis for consolidating existing contributions; it does
not implement a runtime change or designate any PR as approved.

## Decisions requested

1. **Implementation to extend:** should the context work continue from
   [#105297](https://github.com/NousResearch/hermes-agent/pull/105297), with the
   selected continuity work reconciled, or should it be split into a smaller
   follow-up? The recommendation is to build on its context handoff and choose
   one implementation of native-thread binding.
2. **Instruction lifecycle:** are frozen host instructions acceptable initially,
   with identity/skill changes taking effect in a new session, or must the first
   implementation support live instruction revisions? A revision needs defined
   behavior on the next turn, cold resume and native compaction.
3. **Legacy sessions:** should Hermes-only transcripts be explicitly imported,
   or should the initial implementation require a new native session? A lost
   binding for an already native-backed conversation is a different recovery
   problem and must not silently create an empty replacement thread.

## Current behavior and related work

The optional runtime was introduced by
[#24182](https://github.com/NousResearch/hermes-agent/pull/24182). It hands the
agent turn to Codex while Hermes retains interfaces, channel integration,
session storage and background reviews. This differs from using the
`openai-codex` model provider with Hermes' own execution loop.

At the inspected Hermes commit
[`a0749d583a19`](https://github.com/NousResearch/hermes-agent/commit/a0749d583a196f3c7cda94cce596924dec559c27):

- [The app-server branch](https://github.com/NousResearch/hermes-agent/blob/a0749d583a196f3c7cda94cce596924dec559c27/agent/conversation_loop.py#L1502)
  exits before ordinary API-message assembly.
- [The runtime handoff](https://github.com/NousResearch/hermes-agent/blob/a0749d583a196f3c7cda94cce596924dec559c27/agent/codex_runtime.py#L470)
  receives the local message list but calls `run_turn(user_input=user_message)`.
- [Thread creation](https://github.com/NousResearch/hermes-agent/blob/a0749d583a196f3c7cda94cce596924dec559c27/agent/transports/codex_app_server_session.py#L189)
  supplies `cwd` without host instructions. Turn input is text-only; image
  content is reduced to a marker by the adapter's input coercion.
- [Normal message preparation](https://github.com/NousResearch/hermes-agent/blob/a0749d583a196f3c7cda94cce596924dec559c27/agent/turn_context.py#L1018)
  already handles the active system prompt, ephemeral instructions and prepared
  user-context bytes. Those semantics need to survive the native-runtime boundary.

The existing contributions are complementary in some areas and conflicting in
others. States and descriptions below were checked on 2026-09-10:

| Work | Useful contribution | Decision or limitation |
|---|---|---|
| [#26081](https://github.com/NousResearch/hermes-agent/pull/26081), open | First-turn context/history seed | Old runtime integration point; text role labels are not protocol roles; possible duplicate prompt content. |
| [#27998](https://github.com/NousResearch/hermes-agent/pull/27998), open | SOUL forwarding and instruction fields | `baseInstructions` replaces the native base; SOUL is only part of Hermes context. |
| [#74726](https://github.com/NousResearch/hermes-agent/pull/74726), open | Active/ephemeral prompt forwarding chain | Sends `instructions`, which is not the supported thread instruction field. |
| [#72106](https://github.com/NousResearch/hermes-agent/pull/72106), open | Explicit `personality: "none"` | A separate style choice from preserving the native base or delivering host context. |
| [#105297](https://github.com/NousResearch/hermes-agent/pull/105297), open, by s905060 | Context, settings, prepared input, history import and resume | Reconcile prompt contents, instruction changes and persisted binding with other work. |
| [#105502](https://github.com/NousResearch/hermes-agent/pull/105502), open, by JackHunzicker | Scoped continuity, approvals, cancellation, terminal acknowledgement and rich input | Rejects history without a binding; does not itself provide the complete host-prompt handoff. |
| [#103352](https://github.com/NousResearch/hermes-agent/pull/103352), open | Durable binding and strict resume, building on #99012 | Preserve the earlier contribution and its authorship when consolidating. |
| [#98995](https://github.com/NousResearch/hermes-agent/pull/98995), open | Continuity, deadlines, workspace scoping and watchdog behavior | Reconcile unique parts and its proposed recovery behavior with strict continuity. |

The previous consolidation comments in the issue are identified as automated
triage. They do not constitute an approved instruction contract. This proposal
requests a maintainer decision and coordination with the contributors rather
than treating their open implementations as abandoned.

## Proposed ownership contract

Hermes owns product/profile context and its presentation. Codex owns execution,
the authoritative native thread, native tools and native compacted history.

| Content or state | Owner and delivery |
|---|---|
| Native model operating instructions | Codex; leave `baseInstructions` unset in the normal host integration. |
| Stable Hermes host instructions | Hermes; prepare and persist one block, delivered with `developerInstructions`. |
| SOUL, USER, initial memory and applicable channel instructions | Preserve their intended Hermes semantics in that block, subject to existing enablement and context limits. |
| Tool guidance and skill catalog | Describe capabilities actually available to this runtime and session. Preserve explicit user-authored instructions; do not remove content by substring filtering. |
| Project instruction files | Agree on one loader for each source. Prefer native Codex project discovery for its `cwd`, avoiding duplicate AGENTS.md injection and respecting configured exclusions. |
| Retrieved context and plugin data for the current turn | Preserve the existing prepared-input semantics and provenance; do not promote retrieved data into developer authority. |
| User text and images | Map to native input items in order, without duplicating the current user message. |
| Native conversation history | Codex; resume its exact `thread.id`. |
| Hermes transcript projection | Hermes UI, storage and review; not a complete substitute for native state. |
| Native-thread association | One persisted binding, scoped to the owning Hermes session/profile, effective Codex home and working directory. |

Extend existing prompt assembly at its section boundaries. Do not forward a
monolithic prompt and then remove apparent tool instructions with string
matching. Do not add a generic runtime framework solely for this fix.

The capability distinction is material. The current
[`hermes-tools` allowlist](https://github.com/NousResearch/hermes-agent/blob/a0749d583a196f3c7cda94cce596924dec559c27/agent/transports/hermes_tools_mcp_server.py#L41)
excludes `memory`, `session_search`, `delegate_task` and `todo`, while its MCP
instructions advertise some of them. The host block should not introduce
additional instructions to call unavailable tools. Supplying memory text and
exposing a memory tool are separate capabilities.

## Session and update behavior

For an initial implementation, prepare the stable host block once and preserve
its exact bytes across ordinary turns, agent reconstruction and process restart.
Do not reread an edited SOUL file just because a subprocess failed. Existing
profile/context-size rules and channel isolation must still apply.

This frozen-first proposal follows Hermes' conversation cache policy. It does
not resolve genuinely changing per-turn channel overrides; their required
semantics are one of the decisions above. Refreshes at a compaction boundary
also require the native-instruction contract below; observing a compacted event
alone does not update the host block.

When instructions must change within a conversation, all of these properties
need verification:

- The next turn sees the new host instructions.
- Native compaction and cold resume retain the active revision.
- Reapplying the same revision is idempotent.
- Updates do not accumulate unbounded copies of the full host prompt.
- Existing native resources, user/project instructions and unrelated history
  have defined preservation behavior.

Plain `developerInstructions` is not a `turn/start` field in the inspected
protocol. The loaded-thread resume path also ignores instruction overrides.
[#105297](https://github.com/NousResearch/hermes-agent/pull/105297) combines cold
resume with a developer-item injection to address runtime configuration and
retained history. That mechanism needs explicit lifecycle acceptance because a
process restart can affect native resources. This RFC does not claim to have
validated that PR's workaround.

Persist the native binding before admitting user input. Resume the same thread
after reconstruction; a failed or ambiguous resume must not silently start a
new conversation. Use `thread.id`, not `thread.sessionId` interchangeably: the
current protocol can use the latter for a session-tree root.

If transcript import is chosen, make it an explicit operation with a bounded,
role-preserving projection and version-verified `thread/inject_items` support.
Exclude the current user input, foreign encrypted reasoning and untransferable
runtime state. Do not use `thread/resume.history`: the public source marks it
unstable and reserved for Codex Cloud. A transcript import cannot recover
background resources or every element of a lost native conversation.

Preserve the live-thread compaction routing already merged in
[#99000](https://github.com/NousResearch/hermes-agent/pull/99000). Compressing the
Hermes transcript mirror cannot compact Codex's authoritative history.

## Protocol evidence and its limits

A local probe ran the real **codex-cli 0.144.5** app-server against a loopback
mock Responses endpoint on macOS. Each scenario used an isolated configuration
and a synthetic model catalog containing a known base-instruction marker. The
probe captured actual outgoing model requests. It used no model credentials or
real model and did not exercise compaction.

| Scenario | Base marker | Host developer content |
|---|---|---|
| `thread/start(developerInstructions=A)` | Retained | A once |
| `thread/start(baseInstructions=A)` | Replaced | A in the base instead |
| `thread/start(instructions=A)` | Retained | A absent despite accepted RPC |
| Warm resume with `developerInstructions=B`, after a turn with A | Retained | A remains; B absent on the next request |
| Cold resume with `developerInstructions=B`, after a turn with A | Retained | A remains; B absent on the next request |

These results characterize request composition on that version. They do not
establish model compliance, prompt quality, production caching, or behavior of
all Codex releases. The synthetic marker demonstrates preservation versus
replacement without depending on a particular model's complete base prompt.

Relevant upstream evidence:

- [Official App Server documentation](https://learn.chatgpt.com/docs/app-server).
- [Native base selection](https://github.com/openai/codex/blob/537278c65f6b405635de76e91a990105a629110b/codex-rs/core/src/session/mod.rs#L690)
  and [developer-context assembly](https://github.com/openai/codex/blob/537278c65f6b405635de76e91a990105a629110b/codex-rs/core/src/session/mod.rs#L4082).
- [Thread parameter contract](https://github.com/openai/codex/blob/537278c65f6b405635de76e91a990105a629110b/codex-rs/app-server-protocol/src/protocol/v2/thread.rs#L62)
  and [loaded-thread override handling](https://github.com/openai/codex/blob/537278c65f6b405635de76e91a990105a629110b/codex-rs/app-server/src/request_processors/thread_processor.rs#L215).
- [openai/codex#19045](https://github.com/openai/codex/issues/19045), including a
  [separate contributor's 0.153.2 probe](https://github.com/openai/codex/issues/19045#issuecomment-5591093589)
  showing why a visible injected update alone is not necessarily a durable
  compaction baseline. Those compaction results were not rerun for this RFC.

## Delivery and acceptance

After the three design decisions are agreed:

1. Select the existing context contribution and one native-binding implementation;
   reconcile overlapping changes and preserve contributor authorship.
2. Add the agreed host block and prepared-input mapping at the existing runtime
   boundary, preserving normal-runtime behavior and the opt-in setting.
3. Verify first-turn delivery, warm-turn deduplication, reconstruction, cold
   resume, reset/fork behavior and native compaction using actual emitted model
   requests. Check profile/channel isolation and rich input.
4. Preserve per-turn approval revocation, cancellation and exact terminal
   acknowledgement. Do not replay work whose outcome is uncertain.
5. Run the canonical repository test runner and relevant platform checks, then
   a real-model/gateway smoke on explicitly selected supported Codex versions.

Use a small set of parameterized behavioral invariants rather than snapshots
of current field lists. The new implementation tests should fail on the chosen
base and pass after the corresponding change. A successful RPC or mocks that
only check its parameters are insufficient evidence of model-visible context.

Memory/search tool access remains a separate follow-up under
[#26604](https://github.com/NousResearch/hermes-agent/issues/26604). Delegation,
global authentication/config migrations and a general runtime plugin framework
are outside this context RFC. The existing optional runtime remains the target.

## Reproduce the local request probe

Save the Python program below as `probe_protocol.py`, then run:

```sh
python3 probe_protocol.py --codex /absolute/path/to/codex
```

It requires Python 3.11+ and the Codex binary being characterized. It creates
temporary directories, starts a loopback endpoint and isolated app-server
children, and leaves request captures plus a JSON report in the printed artifact
directory. It does not read the user's Codex credentials or personal config.
The mock always returns text, so no model-generated tool calls run. An incompatible
version may fail with a partial report; the program is a diagnostic, not a promise
that unsupported fields will continue to be accepted.

<details>
<summary>Standalone probe source</summary>

```python
"""Local Codex protocol probe: stdio app-server -> loopback mock Responses.

No account credentials, external model requests, or existing configuration.
The synthetic model catalog supplies a known native base prompt so its
preservation can be checked independently from host developer instructions.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = "NATIVE_BASE_CONTRACT_71bd"
A = "HERMES_HOST_A_91ca"
B = "HERMES_HOST_B_48fd"
ROOT = Path(tempfile.mkdtemp(prefix="hermes-codex-contract-"))
CAPTURED = []
BINARY = None


class Endpoint(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CAPTURED.append({"path": self.path, "body": body})
        num = len(CAPTURED)
        item = {"type": "message", "id": f"msg_{num}", "role": "assistant",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}]}
        events = [
            {"type": "response.created", "response": {"id": f"resp_{num}"}},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": {
                "id": f"resp_{num}", "status": "completed", "output": [item],
                "usage": {"input_tokens": 20, "output_tokens": 1, "total_tokens": 21}}},
        ]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Client:
    def __init__(self, root):
        self.root = root
        self.err = (root / "stderr.log").open("ab")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(root),
               "CODEX_HOME": str(root / "codex"), "XDG_CONFIG_HOME": str(root / "config")}
        self.proc = subprocess.Popen([BINARY, "app-server"],
                                     cwd=root, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.err, text=True)
        self.q = queue.Queue()
        self.notes = []
        self.seq = 0
        def read():
            for line in self.proc.stdout:
                self.q.put(json.loads(line))
        threading.Thread(target=read, daemon=True).start()
        self.rpc("initialize", {"clientInfo": {"name": "hermes_contract_probe", "version": "1"}})
        self.send({"method": "initialized", "params": {}})

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def rpc(self, method, params):
        self.seq += 1
        self.send({"id": self.seq, "method": method, "params": params})
        while True:
            msg = self.q.get(timeout=15)
            if msg.get("id") == self.seq:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]
            self.notes.append(msg)

    def turn(self, thread):
        self.notes.clear()
        result = self.rpc("turn/start", {"threadId": thread, "input": [{"type": "text", "text": "Continue."}]})
        turn_id = result["turn"]["id"]
        while True:
            msg = self.notes.pop(0) if self.notes else self.q.get(timeout=15)
            if msg.get("method") == "turn/completed" and msg["params"]["turn"]["id"] == turn_id:
                assert msg["params"]["turn"]["status"] == "completed", msg
                return

    def close(self):
        if self.err.closed:
            return
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=3)
        self.err.close()


def prepare(root, port):
    (root / "codex").mkdir(parents=True)
    catalog = root / "models.json"
    catalog.write_text(json.dumps({"models": [{
        "slug": "contract-probe", "display_name": "contract-probe", "supported_reasoning_levels": [],
        "shell_type": "unified_exec", "visibility": "list", "supported_in_api": True,
        "priority": 1, "support_verbosity": False, "supports_reasoning_summaries": False,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "truncation_policy": {"mode": "tokens", "limit": 10000}, "base_instructions": BASE,
    }]}))
    (root / "codex" / "config.toml").write_text("\n".join([
        'model = "contract-probe"', 'model_provider = "probe"',
        f"model_catalog_json = {json.dumps(str(catalog))}",
        'approval_policy = "never"', 'sandbox_mode = "read-only"',
        '[features]', 'hooks = false', '[model_providers.probe]',
        'name = "OpenAI"', 'wire_api = "responses"',
        f'base_url = "http://127.0.0.1:{port}"', 'requires_openai_auth = false',
    ]))


def counts(stage):
    body = CAPTURED[-1]["body"]
    raw = json.dumps(body, ensure_ascii=False)
    dev = json.dumps([i for i in body.get("input", []) if i.get("role") == "developer"], ensure_ascii=False)
    return {"stage": stage, "native_base_copies": raw.count(BASE),
            "host_A_developer_copies": dev.count(A), "host_B_developer_copies": dev.count(B),
            "host_A_anywhere": raw.count(A)}


def main():
    global BINARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", default=shutil.which("codex"), help="Codex binary to characterize")
    args = parser.parse_args()
    if not args.codex:
        parser.error("Codex is not on PATH; supply --codex /absolute/path/to/codex")
    BINARY = str(Path(args.codex).resolve())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    report = {"codex_version": subprocess.check_output([BINARY, "--version"], text=True).strip(),
              "status": "incomplete",
              "scope": "real local runtime, synthetic catalog and loopback model endpoint", "results": []}
    try:
        for field in ["developerInstructions", "baseInstructions", "instructions"]:
            root = ROOT / field
            prepare(root, server.server_port)
            client = Client(root)
            try:
                response = client.rpc("thread/start", {"cwd": str(root), field: A})
                thread = response["thread"]["id"]
                client.turn(thread)
                report["results"].append(counts(field))
                if field == "developerInstructions":
                    client.rpc("thread/resume", {"threadId": thread, "developerInstructions": B})
                    client.turn(thread)
                    report["results"].append(counts("warm_resume_B"))
                    client.close()
                    client = Client(root)
                    client.rpc("thread/resume", {"threadId": thread, "developerInstructions": B})
                    client.turn(thread)
                    report["results"].append(counts("cold_resume_B"))
            finally:
                client.close()
        report["status"] = "complete"
    finally:
        server.shutdown()
        (ROOT / "requests.json").write_text(json.dumps(CAPTURED, indent=2))
        report["artifact_root"] = str(ROOT)
        (ROOT / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

</details>
