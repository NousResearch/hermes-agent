# Autonomy audit (Phase 0): can NOVA measure, and safely relax, human approval?

Read-only audit before building risk-gated approvals and earned autonomy. Every claim below
is traced to a file and line in this repository, or to TypeSafe's published documentation.
Line numbers are as of commit `40dc9c632`.

**Status (2026-10-02).** §1 is resolved with option (b), and §2 is closed. Both are in
`enforcement.py`, `approvals.py` and `decide.py`, and are tested in
`tests/platform/test_task_approvals.py`.

- **Task work (§1):** an escalated call in a task is filed with its exact arguments in
  `<home>/nova-approvals/<task>/<request>.json`, and the task is held on the board.
  - **release** approves that call once, and **reject** (with a reason) refuses it. Either way the worker runs again.
  - **resume** is refused while a call is waiting.
- **"Always" (§2):** the rule key is now `nova:<action>:<tool_call_id>`. Escalation records
  also carry `tool_call_id` and `session_id`, which is the first half of item 3 below.

**The short answer.** Human approval outcomes *are* observable to NOVA, and can be joined
back to the escalation that caused them, entirely inside NOVA's own plugin — no core patch.
But the audit found three things that change the plan, and they are listed first.

---

## What changes the plan

### 1. In task work, approval never reaches a person

A dispatched worker runs `hermes chat -q`, which marks itself a "single-query" session (`cli.py:4412`). There, the approval gate
does not ask anyone. It reads `approvals.single_query_mode` and either blocks or approves
instantly (`tools/approval.py:894-917`). The default is `deny`
(`hermes_cli/config_defaults.py:1556`, `tools/approval_context.py:260-277`), and NOVA cannot
change it: `approvals` is a reserved key a tenant may not override (`nova/spec/deployment.py:61`),
and NOVA itself only writes `approvals.deny` (`nova/runtime/hermes/materialize.py:413`).

**Consequence:** every NOVA approval action reached from a *task* is refused on the spot, with
no human decision and no approval-outcome hook (the unattended branch fires none). Earned
autonomy therefore has **no human signal at all for task work**. It can only learn from live
chat sessions (gateway or CLI), where a person is actually asked.

**Decision needed:** either (a) scope earned autonomy to chat sessions for now, or (b) first
build a NOVA path that turns a worker's approval request into a board decision (block the
task with the pending call, a person releases or rejects it in the Control Center through the
existing `/work/<id>/decide`, the worker resumes). (b) is real work and its own phase.

### 2. A single "Always" click silently switches a NOVA approval off for good

In a chat session the approval prompt offers **once, this session, and always**
(`tools/approval.py:799-803`, `allow_permanent` defaults on). "Always" is persisted to the
profile's `command_allowlist` keyed by the rule key NOVA sends, `plugin_rule:nova:<action>`
(`tools/approval.py:1016-1019`; NOVA sends `rule_key=f"nova:{action}"`, `enforcement.py:411`). From
then on `is_approved()` returns true **before any prompt or hook** (`tools/approval.py:888-892`,
`315-321`): the action runs ungated, and NOVA hears nothing.

This is a governance hole that exists today, independent of the new features, and it would
also corrupt any agreement metric. **Recommended fix, inside NOVA's plugin:** make the rule key
unique per call (for example `nova:<action>:<tool_call_id>`), so "session" and "always" can never
match a later call — every escalation is asked fresh. (Side effect: "always" still writes one
entry per click to `command_allowlist`; harmless, but it grows.) This should land before
Phase 2, as its own small change.

### 3. "Approved after editing" does not exist in this gate

The person sees NOVA's message and a synthetic label `<tool> (plugin approval rule)`
(`tools/approval.py:1020`), never the tool's arguments, and can only allow or refuse. There is
no edit-then-approve path. What does exist is a **refusal with a reason**: `/deny <reason>` is
relayed as `deny_reason` (`tools/approval.py:812-822`). The ledger's `approved_edited` bucket
should be dropped, or redefined as "refused with a reason, then a revised call approved".

Related: the person approving cannot currently see *what* will be sent. For triage to be
reviewable, NOVA's escalation `message` must include a safe summary of the arguments.

---

## Q1. Where is the human's answer recorded, and can NOVA join it to the escalation?

**The path.** `pre_tool_call` returning `{"action": "approve", "message", "rule_key"}`
(`hermes_cli/plugins.py:1782-1825`) is resolved by `_resolve_block_from_details`
(`plugins.py:1853-1882`), which binds the call's `turn_id`, `tool_call_id` and `session_id`
and calls `request_tool_approval` (`tools/approval.py:1001-1026`) → `_run_approval_gate`
(`870-936`) → `_human_decision` (`730-855`). Note: the "smart" AI auto-approver is **not**
applied to plugin escalations (`_run_approval_gate` calls `_human_decision` without
`smart=True`, `932-936`), so a NOVA escalation always goes to a person or to the
unattended rule.

**Where the answer goes.** Nowhere durable. It becomes the gate's return value and, in the
chat/CLI paths, two plugin hooks:

| Hook | When | Payload |
|---|---|---|
| `pre_approval_request` | before the person is asked | `command`, `description` (NOVA's message), `pattern_key` (`plugin_rule:nova:<action>`), `session_key`, `surface` (`gateway`/`cli`), plus `turn_id`, `tool_call_id`, `session_id` (`tools/approval_context.py:53-72`) |
| `post_approval_response` | after | the same, plus `choice` ∈ `once` \| `session` \| `always` \| `deny` \| `timeout` \| `notify_failed` (`tools/approval_gateway_wait.py:70-75, 158`; CLI `tools/approval.py:843-855`) |

**Can NOVA tell the outcomes apart?**

| Outcome | Observable? | How |
|---|---|---|
| Approved | yes | `choice` ∈ once/session/always |
| Rejected | yes, with a caveat | `choice=deny`; but an interrupted turn (`/stop`, inactivity) is *also* reported as `deny` (`approval_gateway_wait.py:47-50`) |
| Timed out | yes | `choice=timeout` |
| Notification failed | yes | `choice=notify_failed` |
| Approved after editing | **no such outcome** | see "What changes the plan" §3 |
| Auto-approved by an earlier session/always | **not directly** | no hook fires (`approval.py:888-892`); inferable as "escalation followed by the tool running with no `post_approval_response`" |
| Refused unattended (task work) | **no** | no hook; see §1 |
| Who decided | **no** | no actor field in the gateway path (only the smart path sets `decided_by`) |

**Can it be joined back?** Yes. `pre_tool_call` receives `tool_call_id`, `turn_id` and
`session_id` (`plugins.py:1796-1800`); NOVA's hook discards them today (`enforcement.py:349`,
`**_`), and its audit record uses only `HERMES_KANBAN_TASK` or the literal `"runtime"` as the
correlation id (`enforcement.py:316`) — so a chat-session escalation currently cannot be
joined to anything. The approval hooks carry the same `tool_call_id`. **The smallest change:**
NOVA's plugin records `tool_call_id`/`session_id` on its escalation event and registers
`post_approval_response` to append a `policy.approval_outcome` event keyed the same way. This
is NOVA-only (no `CORE_PATCHES.md` entry).

## Q2. What does the hook receive in `args` for approval-gated tools?

**The example's `send_external_email` maps to `email_send`, which does not exist** in Hermes
(`nova/examples/acme/policy.yaml:15-17, 38`; no such tool under `tools/`, `agent/`,
`gateway/`, `plugins/`). That approval can never fire. Hermes's real outbound message tool is
`send_message` (`tools/send_message_tool.py:656-692`):

| Field | Kind |
|---|---|
| `action` (`send`/`list`/`react`/`unreact`) | metadata |
| `target` (`platform:chat_id_or_name`) | metadata (recipient) |
| `message` | **content**; may carry `MEDIA:<local path>` attachments |
| `emoji`, `message_id` | metadata |

Approval-gated tools in real bundles today are MCP tools (HubSpot
`hubspot_batch_update_objects`/`batch_create_associations`/`update_engagement`, Google
Sheets `share_spreadsheet`, `add_rows`, `update_cells`, `batch_update_cells`). Their `args`
are the MCP server's own input JSON, passed through unchanged: record ids and object types
(metadata) beside property values, cell values and sharee emails (**content / personal data**).
Exact per-tool schemas are owned by each MCP server version and should be captured from the
pinned servers before writing `metadata_only` extractors — **not verified here**.

Also: a chat agent's ordinary reply to a customer is model output, not a tool call. No
`pre_tool_call` — and so no approval or triage — ever applies to it.

## Q3. NOVA work decisions vs. the runtime's per-call approval gate

**Two different gates.**

| | NOVA work decision | Runtime per-call gate |
|---|---|---|
| Acts on | a task on the board | one tool call inside a running session |
| Verbs | release, reject, resume, annotate (`nova/runtime/hermes/decide.py:39-56`, `WORK_ACTIONS`) | once / session / always / deny |
| Who | a named Control Center principal; refused without an actor (`decide.py:53-56`) | whoever is in the chat; not recorded |
| Recorded | NOVA audit + the board's event log, same actor (`decide.py:18-19`; `tests/platform/test_work_decisions.py:123`) | nowhere durable (Q1) |

The Control Center's `/decisions` route (`nova/control/api.py:2501-2533`) lists the policy
plugin's `policy.decision` records — refusals and escalations — not human outcomes.

## Q4. Can a worker call TypeSafe, and where does the key live?

- **Network:** the runtime security group allows egress on 443 to `0.0.0.0/0`
  (`deploy/aws/main.tf:276-288`); the instance is in a private subnet reaching Bedrock and ECR
  through NAT, so `api.typesafe.ai:443` is reachable. No proxy is configured.
- **Key:** the agent's own `<profile>/.env`, entered through the Control Center's Credentials
  page — the same path as model keys (`nova/credentials.py`). The runtime loads `.env` into the
  worker's environment, so the standard-library plugin reads `os.environ["TYPESAFE_API_KEY"]`
  (the SDK's own variable name). NOVA never writes the value into a bundle or a compiled file.
- **Caveat:** the triage call would run inside the worker's tool-call path, adding its latency
  to every escalated call; the hook timeout has to be short (the plan's 2 s; the SDK's own
  default is 10 s).

## Q5. TypeSafe: what is documented, and what is not

**Documented** (`https://docs.typesafe.ai/api.md`, `/introduction/quickstart.md`, SDK pages):

- One endpoint: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`,
  JSON. Request: `state` (string, object or array), `model` (`"jev-latest"`), `questions` (a map
  of named questions). All questions go in one call (the documented parallel pattern).
- Questions: `noul` (`instructions`, optional `criteria.true/false`), `choice` (`criteria`: map of
  up to 255 options), `score` (`criteria`: ordered list of 2–10 levels).
- Response: `model` (the **exact version served**, e.g. `"jev-1.13.0"`), `answers`, `usage`
  (`input_tokens`, `output_tokens`).
  - Noul: `{"noul": p}` — **no confidence field.**
  - Choice: `choice`, `probabilities`, `confidence`.
  - Score: `score` is a **probability-weighted value** across levels (e.g. `1.0`), plus
    `probabilities` per level index, `confidence`, and a `legend`.
- Errors (SDK): 400, 401, 403, 404, 422, 429 (with `retry-after`), 5xx, connection error,
  timeout, response-validation error; responses carry `x-typesafe-request-id`. SDK default
  timeout 10 s; env vars `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`.

**Design consequences for Phase 1-2:**

- `min_confidence` can apply only to Choice and Score. For a Noul, the only signal is `p`
  itself; "uncertain" has to be expressed as a band (for example escalate if
  `block_above ≤ p`, and treat `p` in a middle band as uncertain).
- "Score level ≤ `max_allowed_level`" needs a definition. Recommended: escalate unless the
  probability mass at or below the allowed level is ≥ a threshold, rather than reading the
  weighted `score` value.
- The served `model` version is in every response, so "demote on model change" is
  implementable — **if** `jev-latest` can float. Whether a version can be pinned is open.

**Open questions for TypeSafe** (not found in the docs):

1. Rate limits and quotas per key (requests/s, tokens/day).
2. Pricing per token / per question, and whether parallel questions are billed per call.
3. Data retention: are `state` and `questions` stored, for how long, used for training?
4. Processing region(s); EU / Norway processing; a DPA and sub-processor list.
5. Model pinning: can a request name `jev-1.13.0` instead of `jev-latest`, and for how long is a
   version served? How are version changes announced?
6. Latency: typical and p99 for a ~7-question call with a short `state`; any SLA.
7. Calibration: are Noul probabilities calibrated per question type, and is there guidance on
   thresholds for safety gating?

(Note: Jev is also reachable through OpenRouter, per the local `jev` skill. That is a different
data path and contract; this audit assumes the direct TypeSafe API.)

---

## What to do before Phase 1

1. **Decide the scope** of earned autonomy: chat sessions only, or build the board-backed
   approval path for task work first (§1).
2. **Close the "Always" hole** with a per-call rule key (§2) — small, NOVA-only, worth doing
   regardless.
3. **Record outcomes:** extend NOVA's plugin to keep `tool_call_id`/`session_id` on escalations
   and to register `post_approval_response` (Q1). This is the data source the ledger needs.
4. **Fix the example:** map `send_external_email` to `send_message` with `action: send` (Q2).
5. **Get answers** to the TypeSafe open questions, at least retention, region and pinning,
   before any `full_args` mode or any customer pilot.

---

## Live check of the default email questions (2026-10-02, `jev-1.13.0`)

The first live calls through `nova/autonomy/providers/typesafe.py` used the default
`send_external_email` set exactly as the build prompt worded it. That wording escalated an
ordinary shipping note. "Contains personal identifiers" scored 0.40 for a customer's name and
order number (limit 0.10), and "states or promises a date or outcome" scored 0.44 for "it has
shipped" (limit 0.20). With every routine email escalated, the action could never earn
autonomy, so three questions were reworded and checked on the same messages:

| Question | Old wording on routine messages | New wording on routine | New wording on risky |
|---|---|---|---|
| `sensitive_data` | 0.22–0.45 (limit 0.10) | 0.01–0.04 | 0.98–0.99 (card + passport, lab result) |
| `promises_outcome` | 0.14–0.71 (limit 0.20) | 0.03–0.11 | 0.90–0.98 (delivery date, fix deadline, discount) |
| `injected_instructions` | up to 0.07 on how-to text (limit 0.05) | 0.01–0.02 | 0.99 (hidden "system note", bracketed assistant order) |

With the new wording, the full set and the real `combine()` rule gave the expected verdict
on **15 of 15** messages. The 8 routine ones (customer replies, how-to guidance, an internal
note) were `auto_ok`. The 7 risky ones (a guaranteed date, card and ID numbers, a health
detail, a fix deadline, a refund, a prompt injection, and an unsolicited price list to a new
contact) escalated on the question that should catch each one.

**Still open:**

- Latency was 594 ms minimum, 810 ms median and **2688 ms** maximum. Calls slower than
  `timeout_seconds` escalate as `provider_unavailable`, so the 2.0 s default fails safe but
  will cause some needless escalations. Measure p99 in shadow mode before choosing the
  enforce timeout.
- 15 hand-written messages are a check on the wording, not a measure of accuracy. Shadow
  agreement on real traffic is the measure.

### Live check through the installed hook (phase 2)

The example bundle was applied with `autonomy: {provider: typesafe, mode: shadow, data:
full_args}`, and `send_message` was called through the plugin installed in the profile,
which uses its own `_triage.py` and `_triage_typesafe.py`. Both calls were answered by
`jev-1.13.0` in under 1 s. The person was shown what triage found, and the `policy.triage`
records held answers and digests but no message content.

**Finding: the recipient question has nothing to go on.** A real `send_message` call carries
only `target: telegram:<id>`, not who the recipient is. The routine shipping note escalated
on `recipient` alone (confidence 0.86 against 0.90). It passed in the earlier check only
because that check supplied `recipient_type` itself. Before a tenant runs `enforce`, either
give triage a recipient signal NOT taken from the agent's own arguments (for example a
lookup of known customers), or drop the recipient question from the set. Shadow mode will
show how often it is the only failure.
