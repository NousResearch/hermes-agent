# Hermes ACP subagent progress v1

This is a Hermes extension, not a standard ACP subagent protocol. It uses ACP's [documented `_meta` extension mechanism](https://agentclientprotocol.com/protocol/v1/extensibility).

During `initialize`, both peers advertise `_meta.hermes.subagentProgress: 1` in their capability objects. Hermes sends extension metadata only to an opted-in client. Older clients still receive standard `tool_call` / `tool_call_update` notifications titled `Hermes subagent`, with ordinary ACP statuses. A fast child can first appear already terminal.

Every extension carrier has `toolCallId: "hermes-subagent:<id>"`. The carrier's `_meta.hermes.subagentProgress` is a complete snapshot:

```json
{
  "version": 1,
  "id": "child-a",
  "parentId": null,
  "depth": 1,
  "sequence": 1,
  "status": "running",
  "text": "Public assistant output",
  "tools": ["terminal"]
}
```

- `id` and non-null `parentId`: 1–128 ASCII letters, digits, dots, underscores, colons or hyphens. Identity is scoped to the ACP root session. A depth-1 child has `parentId: null`; nested depth is exactly the parent's depth plus one, up to 16. These are identities, never filesystem paths.
- `sequence`: positive monotonically increasing integer within the root journal. Clients ignore duplicate/older snapshots for each child and never revive a terminal child. Nested snapshots can arrive before their parent and wait for it. Parent and depth cannot change for an existing identity.
- `status`: `running`, `completed`, `failed` or `canceled`. Hermes maps `success`/`completed` to completed, `interrupted`/`cancelled`/`canceled` to canceled, and other terminal outcomes (including timeout/error) to failed. Canceled uses ACP's `failed` carrier status because ACP has no canceled tool-call status.
- `text`: cumulative public assistant reply, limited to 16,384 characters and passed through Hermes' existing secret redactor. Pattern-based redaction cannot guarantee removal of all private information. Text is replaced by the latest snapshot, never concatenated during replay.
- `tools`: the first 32 tool-start names, each 1–80 ASCII letters, digits, dots, underscores, colons or hyphens. A tool-start name is activity evidence, not a claim that the tool completed.

No goals, prompts, tool arguments, previews, results, reasoning, completion summaries, model credentials or child-session file paths are sent. The client does not open transcript files. v1 has no child stop/steer API and does not advertise child control capabilities.

Hermes retains at most 64 child snapshots per root session. At these limits later children/activity/text are omitted; this is a bounded visibility window, not the complete conversation archive. Delivery coalesces unsent snapshots per child while retaining the newest cumulative state. Old turn callbacks share the session journal, so detached tasks may report after the root turn finishes or after a subsequent turn begins. A root turn ending does not complete its children.

The journal is stored atomically with mode 0600 under the owning profile's `acp-subagents` directory, using a SHA-256 hash of the server-owned ACP session id. It is replayed during `session/load` and `session/resume`, before the response. The in-process journal keeps running background children live. A cold process restore marks previously running snapshots failed because their liveness cannot be established. Corrupt/oversized journals are ignored. Journals follow the profile's retention; no automatic file pruning is added in v1.

Paseo routes snapshots into its existing read-only provider-subagent panels. Both the built-in generic ACP provider and the plugin SDK use the same validator/reducer. Root history and child replay received before the ACP load response are retained. Unknown versions remain on the standard ACP tool-call path. Closing the runtime cancels remaining visible children; unexpected transport loss fails them. Browser reconnect uses the daemon's retained child timelines.

## Executable verification

`packages/app/e2e/fixtures/hermes-subagents.cjs` is a self-contained NO-MODEL ACP contract fixture. It runs through the actual provider, daemon, WebSocket client and app panels. `e2e/browser/hermes-subagents.spec.ts` covers simultaneous children, nesting, activity, cross-turn terminal states and browser reconnect in desktop light/dark and compact layouts.

For cross-repository verification, set `PASEO_HERMES_ACP_FIXTURE_PYTHON` to the companion Hermes checkout's test interpreter and `PASEO_HERMES_ACP_FIXTURE_SCRIPT` to its `tests/acp_adapter/fixtures/subagent_agent.py`, then run the same Playwright spec. That executable uses the real Hermes ACP server, callback wiring, relay, journal and serializer with a scripted agent. It makes no model requests. All displayed output is explicitly labeled `[NO-MODEL FIXTURE]`.
