# Hermes ACP subagent progress v1

This is a Hermes extension, not a standard ACP subagent protocol. It uses ACP's [documented `_meta` extension mechanism](https://agentclientprotocol.com/protocol/v1/extensibility).

During `initialize`, both peers advertise `_meta.hermes.subagentProgress: 1` in their capability objects. Hermes sends extension metadata only to an opted-in client. Older clients still receive standard `tool_call` / `tool_call_update` notifications with the same projected task title and ordinary ACP statuses. A fast child can first appear already terminal.

Every extension carrier has `toolCallId: "hermes-subagent:<id>"`. The carrier's `_meta.hermes.subagentProgress` is a complete snapshot:

```json
{
  "version": 1,
  "id": "child-a",
  "parentId": null,
  "depth": 1,
  "sequence": 1,
  "status": "running",
  "title": "Count files",
  "text": "Public assistant output",
  "outputLimited": false,
  "activitiesLimited": false,
  "tools": ["terminal"]
}
```

- `id` and non-null `parentId`: 1–128 ASCII letters, digits, dots, underscores, colons or hyphens. Identity is scoped to the ACP root session. A depth-1 child has `parentId: null`; nested depth is exactly the parent's depth plus one, up to 16. These are identities, never filesystem paths. Hermes native relay depth starts at zero; the adapter adds one for this wire contract.
- `title` (optional for older v1 senders): stable, nonempty, single-line public display label, at most 80 characters with no control characters. Hermes first redacts at most 512 characters of the actual goal, then projects it onto a fixed vocabulary of actions and subjects (for example `Count files` or `Review notes`); unmatched goals become `Delegated task`. No arbitrary goal substring is copied. The result is redacted again before egress, saved once, and replayed unchanged. This deliberately loses task detail. Even a task category can be confidential: automatic projection/redaction cannot guarantee that a title reveals no confidential information. Operators must decide whether public delegation visibility is appropriate.
- `sequence`: positive monotonically increasing integer within the root journal. Clients ignore duplicate/older snapshots for each child and never revive a terminal child. Nested snapshots can arrive before their parent and wait for it. Parent and depth cannot change for an existing identity.
- `status`: `running`, `completed`, `failed` or `canceled`. Hermes maps `success`/`completed` to completed, `interrupted`/`cancelled`/`canceled` to canceled, and other terminal outcomes (including timeout/error) to failed. Canceled uses ACP's `failed` carrier status because ACP has no canceled tool-call status.
- `text`: cumulative public assistant reply, limited to 16,384 characters and passed through Hermes' existing secret redactor. Pattern-based redaction cannot guarantee removal of all private information. Text is replaced by the latest snapshot, never concatenated during replay.
- `outputLimited` / `activitiesLimited` (optional additive v1 booleans, default false): sticky indicators set when the 16,384-character or 32-tool-start capacity is reached. Paseo displays an explicit incomplete-record warning once per limit and retains it across reconnect/replay. Reaching a capacity is reported conservatively even if no subsequent content arrives. Text lengths count Unicode code points.
- `tools`: the first 32 tool-start names, each 1–80 ASCII letters, digits, dots, underscores, colons or hyphens. A tool-start name is activity evidence, not a claim that the tool completed.

No raw goals, prompts, tool arguments, previews, results, reasoning, completion summaries, model credentials or child-session file paths are sent. The client does not open transcript files. v1 has no child stop/steer API and does not advertise child control capabilities.

Hermes retains at most 64 child snapshots per root session. Later children/activity/text are omitted. Child snapshots carry the above sticky flags; standard carriers append `[record limited]` to their title for clients that ignore metadata. On the first omitted child, Hermes also sends a completed standard `tool_call` with `toolCallId: "hermes-subagents:limit"` and title `Additional Hermes subagents omitted: 64-child display limit reached`. Its optional `_meta.hermes.subagentLimit` is `{ "version": 1, "sequence": 65, "limit": 64, "childrenOmitted": true }` (sequence is illustrative). Paseo retains this warning in the root timeline. It is replayed before children. Unknown versions use the standard tool-call path. This is a bounded visibility window, not the complete conversation archive; no unbounded omitted-id collection or exact omitted-child counter is retained. Delivery coalesces unsent snapshots per child while retaining the newest cumulative state. Old turn callbacks share the session journal, so detached tasks may report after the root turn finishes or after a subsequent turn begins. A root turn ending does not complete its children.

The journal is stored atomically with mode 0600 under the owning profile's `acp-subagents` directory, using a SHA-256 hash of the server-owned ACP session id. It is replayed during `session/load` and `session/resume`, before the response. The in-process journal keeps running background children live. A cold process restore marks previously running snapshots failed because their liveness cannot be established. The bounded journal object contains `children` and optional `childLimit`; earlier v1 list journals remain readable. Corrupt/oversized journals are ignored. Journals follow the profile's retention; no automatic file pruning is added in v1.

Paseo routes snapshots into its existing read-only provider-subagent panels. Both the built-in generic ACP provider and the plugin SDK use the same validator/reducer. Root history and child replay received before ACP load or resume-only responses are retained. Plugin SDK opening history is capped at 4,096 notifications and 8 MiB, with an opening timeout (default 10 seconds); overflow/timeout fails opening without publishing partial history. Cumulative public-text rewrites replace the child record and change its timeline epoch so reconnect/client caches cannot keep old text. Unknown versions remain on the standard ACP tool-call path. Closing the runtime cancels remaining visible children; unexpected transport loss fails them. Browser reconnect uses the daemon's retained child timelines.

## Executable verification

`packages/app/e2e/fixtures/hermes-subagents.cjs` is a self-contained NO-MODEL ACP contract fixture. It runs through the actual provider, daemon, WebSocket client and app panels. `e2e/browser/hermes-subagents.spec.ts` covers simultaneous children, nesting, activity, safe titles, visible output/activity/child limits, cross-turn terminal states and browser reconnect in desktop light/dark and compact layouts.

For cross-repository verification, set `PASEO_HERMES_ACP_FIXTURE_PYTHON` to the companion Hermes checkout's test interpreter and `PASEO_HERMES_ACP_FIXTURE_SCRIPT` to its `tests/acp_adapter/fixtures/subagent_agent.py`, then run the same Playwright spec. That executable uses the real Hermes ACP server, callback wiring, relay, journal and serializer with a scripted agent. It makes no model requests. All displayed output is explicitly labeled `[NO-MODEL FIXTURE]`.


## Opt-in live-model verification

`tests/acp_adapter/qa/real_model_agent.py` starts the real `AIAgent`, delegate tool and ACP server. It is not a scripted fixture. The companion Paseo `hermes-subagents.real.spec.ts` supplies a temporary home and synthetic workspace, then observes parent completion, a running child, browser reconnect and terminal public output. Set `PASEO_HERMES_REAL_PYTHON`, `PASEO_HERMES_REAL_SCRIPT` and `PASEO_HERMES_REAL_AUTH_HOME` to the test interpreter, this QA script and an already authenticated Codex profile, then run the spec using Playwright's `real-provider` project. The official credential resolver is called with `read_only=True`; credentials stay in memory. Authentication and QA homes must not overlap. Normal authentication must be completed beforehand; this runner never refreshes or writes the authentication store.

Set `PASEO_HERMES_REAL_FAIL_CHILD=1` for a separate failure-injection run. Only the isolated QA configuration pins the child to an intentionally unavailable model, with no fallback; the parent still uses the real authenticated model. This induces a genuine child runtime failure through normal model routing, not successful child inference or an organically occurring failure. The UI test records the observed failed state, without forwarding the raw upstream error body. No synthetic progress events are injected. Live model timing and tool use are nondeterministic; this opt-in test makes network requests and may consume quota. It is separate from deterministic no-model CI coverage.
