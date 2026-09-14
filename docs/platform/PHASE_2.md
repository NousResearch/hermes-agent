# Phase 2 — Policy & Governance

Business-action controls that are actually enforced, not recorded and hoped for. Phase 1
carried `permissions` and `approval.required_for` through to the runtime as inert
declarations; Phase 2 makes them decide what an agent can do.

**Core patches spent: still 1** (the `AGENTS.md` routing row). Enforcement rides a
documented runtime extension point.

---

## How enforcement actually works

The runtime exposes `pre_tool_call`, which its own source calls "the policy hook". A
plugin returning `{"action": "block", …}` vetoes a tool call; returning
`{"action": "approve", …}` escalates it to **the same human gate that guards dangerous
shell commands** — a gate that fails closed when no human is present.

NOVA compiles each agent's policy and installs a plugin into that agent's profile:

```
<profile>/
  nova-policy.json              the compiled policy for this agent
  plugins/nova-policy/
    plugin.yaml                 manifest declaring the pre_tool_call hook
    __init__.py                 the hook
    _decide.py                  verbatim copy of nova/policy/decide.py
```

A dispatched worker runs with its profile as the runtime home, and plugin discovery scans
`<home>/plugins` — so installing here scopes the policy to exactly one agent, with no
shared state between agents on the same host. The same tool gets opposite answers for
different agents, which is the property a security review will test first.

### One decision function, two places

`nova/policy/decide.py` is copied verbatim into every agent's plugin directory. The code
that explains a decision in the control plane and the code that enforces it inside a
worker are the same code, byte for byte, and a test asserts it.

Two implementations of a security decision will eventually disagree, and the disagreement
will be discovered by a customer.

### Precedence

Evaluated in this order, and **deny is strongest**:

| # | Rule | Effect |
|---|---|---|
| 1 | Tool is in the agent's `tools.deny` | **deny** |
| 2 | Tool is in the tenant baseline | allow |
| 3 | Tool performs an action requiring approval | **escalate to a human** |
| 4 | An allow-list is in force and the tool is granted | allow |
| 5 | An allow-list is in force and it is not | **deny** |
| 6 | Otherwise | the tenant's `unlisted_tool` default |

An explicit denial beats the baseline. A customer who writes `deny: [terminal]` means it;
if that breaks a worker, the **compiler warns at build time** rather than the runtime
overriding the customer at execution time.

### Failing closed

Every failure path blocks: a missing policy document, a corrupt one, an unsupported
schema version, or an exception inside the decision itself. A control that fails open is
not a control. A worker that is too restricted fails loudly and visibly; one that is
silently unrestricted does not.

---

## Declaring policy

`policy.yaml` in the tenant bundle. Business actions are named in the customer's language,
because the person who decides that refunds need approval is not the person who knows
which tool issues one.

```yaml
actions:
  refund:
    description: Issue a refund to a customer
    tools: [crm_refund]
    requires_approval: true          # tenant-wide default

permissions:
  read_customers:
    tools: [crm_lookup, crm_search]

baseline_tools:                      # what every worker keeps, whatever its permissions
  - kanban_complete
  - kanban_block
  # …

defaults:
  unlisted_tool: deny                # the posture a security review will ask for
```

An agent then draws on it:

```yaml
permissions: [read_customers, create_ticket]
approval:
  required_for: [send_external_email]   # this agent escalates more than the tenant default
tools:
  deny: [terminal, execute_code]
```

**A bundle that references an undefined permission or action fails to load.** A control
that grants nothing, or an approval requirement nothing can trigger, is the worst kind of
governance failure: it looks present in a review and does nothing.

### This resolves the Phase 1 limitation

Positive tool scoping did not ship in Phase 1 because the only available mechanism would
have stripped the tools a worker needs to report task completion — producing work that
runs and never closes. The baseline concept solves it: an allow-list narrows what an agent
can reach while the baseline guarantees it can always close its own task.

---

## Control API

Three routes added, all read-only.

| Route | Returns |
|---|---|
| `GET /policy` | The tenant policy and what it compiles to per agent, with an `enforced` flag |
| `GET /policy/simulate?agent=&tool=` | What policy would do, and why — using the enforcement function itself |
| `GET /decisions?agent=&limit=` | Refusals and escalations the runtime recorded |

`simulate` answers the question a customer's security team actually asks — *"what happens
if this agent calls that?"* — and a test asserts its answers are identical to
`decide()`'s, so the explanation cannot drift from the behaviour.

`/decisions` deliberately excludes permitted calls. They are the overwhelming majority and
would bury what a reviewer is looking for. What was refused, and what needed a human, is
the governance question.

---

## Governance record

Every refusal and escalation is appended to the tenant audit log as a `policy.decision`
event carrying the agent, tool, effect, rule and a human-readable reason. Written by the
plugin from inside the worker process, into the same log the platform writes to, so there
is one record rather than two.

A record that cannot be written never stops the decision being enforced — the enforcement
matters more than its paper trail.

---

## Known limitations

- **Approval decisions are made by the runtime's gate, not by NOVA.** NOVA decides *what*
  needs approval and escalates it; the human answers through the runtime's existing CLI
  prompt or chat round-trip. Building a second approval UI that does not actually gate
  anything would be worse than none.
- **Policy applies to tool calls, not tool arguments.** "Refunds over £500 need approval"
  is not expressible yet; "refunds need approval" is. Argument-level predicates are a
  later phase.
- **An agent with no permissions runs without an allow-list.** Deliberate: introducing
  governance must not silently restrict agents that predate it. Set `unlisted_tool: deny`
  and grant permissions explicitly to close that.
- **The plugin is installed only when a policy is declared.** Bundles without `policy.yaml`
  behave exactly as they did in Phase 1.
- **One tool performs one business action.** Enforced at load; a tool claimed by two
  actions is a declaration error.
- **No per-agent rate or spend limits.** `limits` in an AgentSpec is still recorded, not
  enforced. Cost control is its own phase.

---

## Extension points

**Add a rule type.** Extend `decide()` in `nova/policy/decide.py` and the compiled document
it reads. It is copied into the runtime on the next apply, so enforcement follows
automatically — but keep it dependency-free, because it runs where NOVA is not installed.

**Add an enforcement point for another runtime.** Implement the adapter's own installation
step and reuse `nova/policy/decide.py` unchanged. Report the result through
`RuntimeCapabilities.policy_enforcement`; an adapter that cannot enforce must say so, and
`apply` refuses to present governance that does not exist.

**Add a governance event.** Record it through the audit log with a new `kind`. Policy
decisions are not model-visible changes and use `record()`, not the write-ahead guard.

---

## Tests

`tests/platform/` — 228 tests total, 58 of them Phase 2.

| File | Covers |
|---|---|
| `test_policy_decide.py` | Precedence, failing closed, reasons on every decision |
| `test_policy_model.py` | Declaration validation, compilation, warnings, digest identity |
| `test_policy_enforcement.py` | The installed plugin, loaded from disk with NOVA absent |
| `test_control_api.py` | Policy routes, simulate/enforce equivalence, decision records |

The enforcement tests import the plugin **by file path with NOVA off the import path**. If
they passed only because `nova` happened to be importable, they would not be testing what
ships.
