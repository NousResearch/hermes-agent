# Employee tool surface

Status: tool decisions agreed; implementation pending.

This document defines the model-visible tool surface of this Hermes fork.
Implementation follows [the master specification](employee.md); this document
records contracts rather than claiming they are already implemented.

## Agreed decisions

| Area | Intended surface |
| --- | --- |
| Browser | `browser_exec`, rather than the separate navigation/click/type browser tools. |
| Skills | Remove the skills primitive from the employee experience, including `skills_list`, `skill_view`, and `skill_manage`. This decision does not require deleting unrelated upstream internals wholesale. |
| Scheduling | Author schedule declarations through responsibility files using ordinary file tools. No standalone scheduling tool. Native cron remains the execution mechanism. |
| Deeper memory | Hindsight, using the established employee recall/retention mechanism and `recall` surface. Do not substitute a different memory backend or interaction pattern. Exact integration and configuration will be specified separately. |
| Messaging | Expose `send_message` to the model, including scheduled work. |
| Issue reporting | Do not expose `report_issue`. |
| Video understanding | Expose `video_analyze`, backed by Gemini through OpenRouter. |
| Extra tools | No Kanban, clarification tool, computer-use tool, home-automation tools, or text-to-speech tool in the employee surface. Asking a question in ordinary conversation remains possible. Other optional/desktop/plugin tools are not automatically admitted; add only through an explicit surface decision. |
| Service access | Expose neither `connection` nor `manage_connections`. Do not use the Nous connector gateway. Service access uses native CLI logins, configured secrets and MCP; local MCP support remains independent of the removed setup tool. |

Existing shared working tools (terminal/process management, file operations,
web research, image analysis/generation, todos, authored memory, session search,
code execution and delegation) are the comparison baseline. Their detailed
schemas and behavior still need comparison; matching names alone are not proof
of parity. No additional optional tool is approved merely by its availability
upstream.

## Preserve tool behavior, not just names

Reuse the established employee tool implementations and model-visible contracts
where they match these decisions. This includes descriptions, arguments, result
shapes, validation, limits, warnings and actionable error messages. Prefer
reusing applicable code directly in native owners over recreating its behavior
or adding hosted interfaces with local adapters.

Adapt paths, runtime facts and references to removed capabilities. Do not copy
messages instructing the model to use `connection`, `report_issue`, hosted
request links, or nonexistent knowledge locations. Service manuals remain,
independently of credential storage, under [the agreed guide contract](guides.md).

### File tools

Decision: preserve the established responsibility-file limits and feedback
behavior exactly, including read usage gauges and linked-file listings, write
and patch validation, and consolidation/repair warnings. Adapt only local paths
and references to capabilities explicitly excluded from this fork. Do not tune
budgets or simplify this feedback during the port.

Preserve the responsibility-aware behavior across `read_file`, `write_file`
and `patch`, including package validation, linked-file discovery, usage gauges,
and warnings that teach the model how to repair rejected writes. Limits apply
to the governed knowledge files, not to arbitrary user documents or source code.
The established limits to carry forward with the corresponding package layout:

| File / collection | Limit |
| --- | --- |
| `RESPONSIBILITY.md` | 12,000 characters |
| `STATE.md` | 6,000 characters |
| Each `state/` Markdown file | 10,000 characters; 100 files |
| Each `archive/` Markdown file | 130,000 characters; 500 files |
| Each `references/` Markdown file | 10,000 characters |
| Each package script | 128 KiB |
| Each schedule/webhook declaration | 16 KiB, when that declaration type is supported |

Preserve the distinction between successful writes with advisory warnings and
rejected mutations. An oversized handoff should receive consolidation guidance;
a malformed charter should explain the required fields; a finite responsibility
must state its completion condition. Apply the same validation to patch results,
not only whole-file writes. Retain applicable read behavior for existing files
that require consolidation rather than making their contents inaccessible.

Preserve the reference warning wording except for the local-path and excluded-
capability adaptations above. Other tools and unrelated runtime restrictions
remain subject to their own surface decisions. Terminal edits bypass file-tool validation,
so do not describe these limits as an OS-enforced filesystem boundary.

### Messaging

Decision: preserve the established employee `send_message` contract as seen by
the model: tool description, argument schema, recipient discovery, results,
errors and delivery feedback. Use native Hermes delivery underneath; that
implementation choice must not silently change the model-facing contract.
Automatic conversation replies remain distinct from explicit tool sends.
Expose the tool in ordinary employee conversations and scheduled runs.

Model-visible parity is the acceptance criterion for this port, subject only
to the local-environment adaptations and capability exclusions already agreed.
If a reference result promises behavior the native delivery path does not yet
provide, implement the required behavior or surface the discrepancy for a
decision; do not merely copy the promise into the description.

### Authored memory and personal profiles

Decision: preserve employee-style `memory` behavior, separately from the
already-selected Hindsight recall mechanism.

- `target="memory"` stores shared organization facts, with a 2,200-character
  budget. This block is frozen into the conversation's system prompt.
- `target="user"` stores a particular person's identity, role, preferences
  and working style, with a 1,375-character budget per person. It is not one
  shared `USER.md` for everyone using a Hermes profile.
- Authored turns default to the user target; unattended runs without a bound
  person default to shared memory. Shared conversations require the `user`
  selector naming the recorded sender label. Preserve attribution validation
  and corrective errors for missing, unknown or ambiguous selectors.
- Preserve add/replace/remove, atomic batches checked against the final budget,
  duplicate handling, usage feedback and consolidation errors.
- Use local persistence with stable person identity and profile isolation.
  Storage adaptation must preserve the model-facing behavior.

#### Exact context placement

Personal memory is **not at the top of context and not in the system prompt**.
Load the current speaker's profile fresh for each turn and append it after the
current user message's original content in the API-bound copy, in this order:

```text
<original current user message>

<user-profile-context>
[System note: The following is persistent user-profile memory, NOT new user input. Treat as authoritative reference data about the current user.]

<current speaker's personal memory>
</user-profile-context>

<memory-context>
<Hindsight recalled context, when present>
</memory-context>

<additional plugin user-message context, when present>
```

For multimodal messages, preserve the original text/image blocks and append the
ordered context as a final text block. Do not insert a synthetic user turn.
Keep stored user content clean and persist the exact API-bound context for
subsequent replay; never reload personal memory into historical messages.
A personal-memory update therefore affects that person's next turn without
rewriting the conversation's cached prefix. Other speakers do not receive that
person's profile as their own current-user context.

Preserve this ordering and wrapper wording through actual request assembly,
including session restoration. Do not infer from this decision that unrelated
reference-runtime prompt refresh policies are approved.

## Native service access: current behavior

Hermes has several independent mechanisms:

- **Managed app accounts:** `manage_connections` supports status, connect and
  reconnect through the Nous tool gateway. Interactive CLI/TUI/desktop sessions
  show connection cards; channels without cards receive authorization links.
  Actual app operations use gateway connector tools. Disconnect/revoke is a
  user action, not a model-tool action.
- **Local MCP:** the same tool also supports install, enable and authorize for
  MCP catalog targets explicitly marked `mcp: true`. MCP tools are used through
  native discovery/dispatch. This is distinct from the hosted app-account route.
- **Availability:** `manage_connections` is gated by connector configuration and
  gateway/account availability (including the supported guest state). Its mixed
  interface is not an unconditional local credential manager.
- **CLI/API access:** tools can use installed service CLIs and their native
  login state, or configured API credentials. This does not require an employee
  credential-request tool.
- **Secret sources:** profile `.env` and native secret sources support credentials;
  bundled sources include Bitwarden, 1Password and a command source. Credential
  loading and terminal delivery have explicit scope rules; fetching a secret
  does not mean every terminal process automatically receives it. These mechanisms
  do not provide the employee broker's placeholder-substitution guarantee.

Implementation references: [connection tool](../../tools/connectors/tool.py),
[availability gate](../../tools/connectors/gateway/config.py),
[secret-source registry](../../agent/secret_sources/registry.py), and
[secret configuration](../../website/docs/user-guide/secrets/index.md).

Decision: exclude `manage_connections` because this fork will not use the Nous
connector gateway. Retain native CLI authentication, configured secrets and
MCP support without this model-facing setup tool. Do not introduce a replacement
connection tool. Adapt any model-visible guidance that otherwise tells the
agent to call `manage_connections`; removing it must not disable independently
configured local MCP servers.

## Implementation checks

Verify the actual assembled model schema, not just a static toolset list.
Removed tools must not reappear through platform defaults, plugin discovery or
review/delegation paths. New sessions get the selected surface without mutating
an existing conversation's cached prefix. Verify real file-tool validation and
feedback, native send execution, and Hindsight retention/recall behavior with
profile isolation. Record runtime divergences in the downstream ledger when
implemented; this specification is not evidence that those changes have landed.

## Telegram transport and display

Decision: retain native Hermes Telegram tool-progress configuration and display
behavior. Prefer the existing Telegram adapter and channel connection setup;
do not replace them with the hosted implementation. Implement employee behavior
in shared prompt, memory, file-tool and scheduling owners. Use native sender and
session metadata for person attribution. Any adapter change requires a concrete
missing contract demonstrated by tracing the native path, not assumed parity
work. Employee conversation additions follow [conversation framing](conversation-framing.md).
Display defaults follow [runtime defaults](runtime-defaults.md).

## Additional native behavior decisions

- Attachments retain native Hermes handling and presentation.
- Delegation retains native Hermes behavior, including child execution and
  completion handling; do not port a separate hosted delegation path.
- Mid-turn steering retains native Hermes behavior and framing.
- For remaining tool descriptions, results, warnings and errors, prefer the
  employee behavior where it serves the agreed product. Apply judgment rather
  than copying every difference: keep behavior consistent with actual native
  execution and excluded capabilities. Previously explicit parity decisions
  (including responsibility file limits and messaging) still apply.
- Shared-conversation framing follows [the agreed framing contract](conversation-framing.md):
  native assembly with person-aware labels, compact session references and
  confirmed outbound-delivery context. Do not copy hosted channel adapters.
