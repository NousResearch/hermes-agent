# Earned autonomy: risk-gated approvals that agents earn

> Agents that earn trust like a new employee: supervised first, independent once proven,
> with proof.

This document is the design, the threat model and the operating guide for two connected
features:

- **Risk triage.** An approval-gated tool call is checked by a decision provider's answers
  to a fixed set of questions. A rule in code combines those answers.
- **Earned autonomy.** An action starts supervised. NOVA measures how often people agree
  with triage. When an action proves itself, NOVA proposes promoting it, and an
  administrator confirms. Any sign of trouble demotes it automatically.

The audit that came before it, with the provider's API and the open questions, is
[`AUTONOMY_AUDIT.md`](AUTONOMY_AUDIT.md).

---

## 1. What can relax an approval, and only that

Triage is a **second step** that runs only when `decide()`
(`nova/policy/decide.py`) returned `REQUIRE_APPROVAL`. It never sees a call that policy
denied, a baseline tool, a call over the per-run ceiling, or a call outside the
allow-list. All of those are decided before it, and its tests prove they are unchanged
byte for byte. The most triage can ever do is let **one escalated call** run. It does that
only when **every** condition below holds:

| Condition | Where it is checked |
|---|---|
| The tenant chose `mode: enforce` | compiled policy, `enforcement.py::_triage` |
| The action is `graduated` | compiled policy, then **re-read from disk** before acting (`_still_graduated`) |
| The provider is a real one, not `fake` or `none` | refused at load (`nova/autonomy/spec.py`) and again in the hook |
| The model answering is the one the action graduated on | `_triage`, using the model version in every answer |
| Every question passed its configured threshold | `nova/autonomy/triage.py::combine` |

Anything else gets the escalation that would have happened without triage, with what
triage found added to the message the person reads. In `shadow` mode, the default, every
call escalates and the person sees "Triage (shadow) would have let this through" or
"would have asked a person anyway — …".

A call that runs without a person writes a `policy.autonomous_action` **intent before it
runs**. The runtime's `post_tool_call` hook then writes `committed` or `failed`. A worker
that dies in between leaves an open intent, which is how a reviewer finds an action whose
outcome is unknown.

## 2. How the pieces fit

```
tool call ─▶ decide() ──allow/deny──▶ (unchanged)
                │
        require_approval
                │
                ▼
     autonomy configured for this action?
           │ no ─────────────▶ escalate (unchanged)
           │ yes
           ▼
   build state (metadata_only | full_args, secrets redacted)
           ▼
   provider.ask(state, questions)  ── hard deadline, no retries
           ▼
   combine(questions, answers)     ── rule in code, thresholds in the compiled policy
           ▼
   enforce + graduated + same model + auto_ok + still graduated on disk?
           │ yes ─▶ intent ─▶ tool runs ─▶ committed/failed
           │ no
           ▼
   escalate with the triage note
     chat:  the runtime's prompt, rule key unique to the call
     task:  the exact call filed on the board, the task held for a person
           ▼
   the person's answer ─▶ policy.approval_outcome (chat) / approvals store (board)
           ▼
   ledger ─▶ assess ─▶ proposal (admin confirms) / demotion (automatic)
```

| Part | File | Runs in |
|---|---|---|
| Question model, default sets | `nova/autonomy/questions.py` | control plane |
| The combining rule, state building, redaction | `nova/autonomy/triage.py` | **both**: copied into each profile as `_triage.py` |
| Provider wire client | `nova/autonomy/providers/typesafe.py` | **both**: copied as `_triage_typesafe.py`, standard library only |
| Fake provider | `nova/autonomy/providers/fake.py`, `triage.safe_answers` | both |
| Policy block, promotion and demotion rules | `nova/autonomy/spec.py` | control plane (compiled into `nova-policy.json`) |
| Hook integration | `nova/runtime/hermes/enforcement.py` | worker, gateway |
| Ledger | `nova/autonomy/ledger.py` | control plane |
| Promotion and demotion | `nova/autonomy/review.py`, `nova/autonomy/state.py` | control plane |
| Routes | `nova/control/autonomy.py` | control plane |
| Screen | `nova/control/ui/src/screens/autonomy.tsx` | browser |
| Report | `nova/autonomy/report.py`, `nova autonomy report` | CLI |

## 3. Configuring it

In the tenant's `policy.yaml`:

```yaml
autonomy:
  provider: typesafe          # typesafe | fake | none   (none = off)
  mode: shadow                # shadow | enforce
  data: metadata_only         # metadata_only | full_args
  timeout_seconds: 2.0        # 0.2 – 10; slower answers escalate
  min_confidence: 0.90        # optional; overrides the default sets' Score/Choice minimum
  actions:
    send_external_email:
      questions: default      # or an inline list of noul / score / choice questions
  promotion:                  # defaults shown
    min_shadow_decisions: 50
    min_agreement: 0.98
    max_false_safe: 0
    window: 100
  demotion:
    on_rejection_of_autonomous: true
    on_false_safe_rate_above: 0.02
    on_provider_model_change: true
```

These are refused when the bundle loads, each with a sentence that names the field:

- `enforce` with the `fake` or `none` provider;
- an action that is not a declared business action;
- a threshold outside 0–1;
- an allowed option that is not an option;
- a level that does not exist;
- a duplicate question id;
- a timeout outside its bounds;
- a promotion window smaller than its minimum;
- a `graduated` action without the model version it graduated on.

**State is not written by hand.** An action's state lives in `autonomy_state.yaml` in
the bundle. Only a confirmed promotion or a demotion writes it, through the validated
bundle writer, audited `intent → committed`. That file is kept apart from `policy.yaml`,
so a machine write never rewrites the file a person wrote. A damaged state file reads as
"supervised".

The provider key is the agent's own credential (`TYPESAFE_API_KEY`), entered on the
Control Centre's Credentials page like a model key. NOVA never writes it into a bundle or a
compiled file.

### The questions

Each question is atomic and has its own threshold:

- **Noul:** the probability a statement is true. It fails at `p ≥ block_above`. The
  provider gives no confidence for a noul, so the threshold is the confidence bar.
- **Score:** a rating on ordered levels. It fails if the likeliest level is above
  `max_allowed_level`, if confidence is below `min_confidence`, or if the probability
  of being *at or below* the allowed level is under `min_confidence`. The weighted score
  is not used, because it can look safe while real probability sits at the top level.
- **Choice:** one of the options. It fails outside `allowed`, below `min_confidence`, or
  when the probability of an allowed answer is under `min_confidence`.

The default `send_external_email` set (`questions.py::DEFAULT_SETS`) has:

- five nouls: financial commitment, sensitive personal data, a promised date or outcome,
  professional advice, and injected instructions;
- a sensitivity score;
- a **recipient** fact, allowed only for `existing_customer` and `internal`.

Three of the nouls were reworded after a live check showed the first wording escalated
ordinary customer email (see the audit doc).

### The recipient is looked up, not asked

Who a message goes to is a fact, not a judgement. A provider shown `telegram:42` can only
guess from the tone of the message, and live it answered with 0.86 confidence on every
routine email, which escalated all of them. The agent cannot be asked either, because it is
the party being checked. So the `recipient` question type is answered by NOVA itself, in
`triage.py::recipient_answer`, from two records the agent cannot write:

1. **The tenant's contact list.** `contacts.yaml` in the bundle, edited under Settings →
   Contacts (admin-only, because it holds real people's chat ids). It is compiled into the
   policy of agents whose triage asks about the recipient, and only theirs.
2. **The gateway's channel directory.** `channel_directory.json`, rebuilt by the gateway
   every few minutes from real conversations, outside every workspace.

| Found | Answer | Passes by default |
|---|---|---|
| On the customers list | `existing_customer` | yes |
| On the internal list | `internal` | yes |
| Not listed, but a direct chat that has written to us | `known_contact` | **no**. A stranger can write first, so "has written to us" is not "is a customer". A tenant can allow it in an inline set. |
| A group or channel | `broadcast` | no |
| Not found, no target, ambiguous name, unreadable files | `unknown` | no |

A friendly name (`telegram:Lena`) is resolved through the directory, but only when exactly
one chat has that name. A recipient type in the call's own arguments is ignored. The
provider is never sent the question. The audit records the answer and where it came from,
never the chat id.

## 4. What leaves the environment

| Mode | What the provider receives |
|---|---|
| `metadata_only` (default) | The tool name, the business action, and the platform (`telegram`, not the chat id). Per argument: its type and size signals (characters, lines, links, email addresses, digits, attachments). **No value of any argument.** |
| `full_args` | The arguments themselves. The screen shows "Content leaves your environment to the triage provider". |

In **both** modes, before anything is sent:

- strings shaped like credentials are replaced: provider keys, GitHub, Slack, AWS and
  Google keys, Telegram bot tokens, JWTs, bearer tokens, private keys, and
  `password=`/`token=` pairs;
- the value of any argument whose name looks like a secret is replaced;
- **every value of the worker's own secret-named environment variables** (its `.env`) is
  replaced, whatever it looks like.

The audit log never holds content. A `policy.triage` record keeps the answers,
probabilities, verdict, failed questions, provider, model version, latency, tokens and data
mode, plus **digests** of the state and the arguments.

`metadata_only` is the safe default and has an honest cost. Most content questions cannot
be answered from size signals, so triage rarely calls an email safe in this mode. To
measure a content-based action, a tenant has to choose `full_args` knowingly.

## 5. The ledger, promotion and demotion

The ledger joins each triage verdict to the person's answer to the same call:

- **Chat:** the plugin's `post_approval_response` hook writes `policy.approval_outcome`,
  keyed by the call's `tool_call_id`.
- **Board:** the approvals store records who released or rejected the held call, keyed by
  the request id the triage record also carries.

**Shadow agreement** is the share of safe verdicts a person reviewed and approved. It is
what justifies a promotion. **False-safe** is a safe verdict a person rejected. It is shown
first on the screen, in the warning colour.

**Promotion is proposed, never applied.** An action qualifies only when all of these hold:

- its window holds at least `min_shadow_decisions` reviewed safe verdicts;
- agreement is at least `min_agreement`;
- there are no more than `max_false_safe` false-safe verdicts;
- **all** of those decisions were answered by one model version, the current one;
- the provider is real.

NOVA records `autonomy.promotion_proposed`. An administrator confirms on the Autonomy
screen, and eligibility is checked again at that moment. The confirmation is an audited
bundle write followed by an apply, so the compiled policy says `graduated` with the model
version.

**Demotion is automatic.** Any of these sends a graduated action back to supervised, through
the same audited write:

- a person reports a wrong call the action made without them;
- the false-safe rate rises above `on_false_safe_rate_above`;
- the provider starts answering with another model.

The hook also refuses a different model on its own, immediately. It re-reads the compiled
policy before acting without a person, so a long-lived gateway process that loaded the
graduated policy asks a person from its next call.

**Reviews run** whenever the Autonomy screen is read, at once on a reported incident, and
with `nova autonomy review <bundle>`. Schedule that command (for example every 15 minutes)
so a demotion never waits for someone to open a page.

## 6. Threat model

| Threat | What could happen | What stops it |
|---|---|---|
| **Prompt injection in content** | Text in a message the agent sends tries to steer the triage model into calling it safe, or the agent itself was steered into sending it. | A dedicated injection question with the strictest threshold (0.05; real attacks scored 0.99 live). The model never decides alone: a single failed question escalates. Thresholds live in NOVA, not the provider. `metadata_only` sends no content to inject. Graduation needs a sustained record of people agreeing, and one incident demotes. |
| **Prompt injection in the triage request** | Content is placed inside the state sent to the provider. | The questions and thresholds are fixed and separate from the state. An injected "answer: safe" in the state cannot change a threshold, and a malformed answer escalates. |
| **Provider outage, slowness, error** | 401, 422, 429, 529, 5xx, timeouts, an unreachable network. | Every failure is "no answer", which escalates. A hard deadline covers the whole call, with no retries inside the hook. |
| **Provider changes its model** | `jev-latest` moves to a new version with different calibration. | The model version is in every answer. A graduated action only proceeds on its own version, the hook refuses others at once, and review demotes. Trust is earned again from zero. |
| **A wrong-but-confident provider** | Systematic false-safe verdicts. | Shadow first. Promotion needs ≥ 98% agreement and zero false-safes over the window. False-safe rate demotes. People can report an incident, which demotes on the spot. |
| **Data residency** | Content sent to a third party, possibly outside the EU or Norway. | `metadata_only` by default, `full_args` opt-in and shown on screen, secrets redacted in every mode. The provider's processing region and retention are **open questions** (§9). |
| **Self-promotion by an agent** | An agent edits its own policy or state to graduate. | `autonomy_state.yaml` is in the bundle, outside every workspace. The profiles directory is never writable by file tools, and the compiled policy is re-read from a path an agent cannot write. As with every NOVA control, an agent granted an unrestricted shell is outside this boundary; deny `terminal` to agents whose actions are triaged. |
| **Replaying an approval** | A task approval is reused for a different call. | The board grant is keyed to the exact call (tool + canonical arguments) and spent by an atomic rename, so it lets one call through once. |
| **"Always" in chat** | One "always" click silently pre-approves every later call. | The rule key is `nova:<action>:<tool_call_id>`, unique per call, so a saved "always" never matches another call. |
| **Viewer or cross-site write** | Someone without the right promotes an action. | Promote, demote and incident are admin-only, and the API refuses a viewer, not just the UI. The server rejects cross-origin writes. |

## 7. Fail-closed table

| Failure | Result |
|---|---|
| `autonomy` absent, or `provider: none` | No triage. Behaviour is identical to before, proven by a test. |
| Triage files missing from a profile (old install, gateway default profile) | No triage: the escalation goes to a person |
| No API key | Escalate, `provider_unavailable (no API key …)`, and no request is made |
| HTTP error, timeout, unreachable, non-JSON, undocumented shape | Escalate, `provider_unavailable (…)` |
| Missing answer, wrong type, probability outside [0, 1], NaN, missing model version | Escalate |
| Any answer fails its threshold | Escalate, listing every failed question and why |
| Model differs from the graduation model | Escalate |
| State file damaged, or graduated without a model version | Read as supervised |
| Compiled policy unreadable at the moment of acting | Escalate (not shown to be still graduated) |
| Exception anywhere in triage | Escalate with "Triage could not run" |
| Worker dies after an autonomous call starts | Open `policy.autonomous_action` intent, visible to review |
| Crash during a promotion or demotion write | Open intent in the audit log |
| Audit log not writable | A refusal or escalation is still enforced (its record is lost). A call that would run without a person **escalates instead**: no intent, no autonomous call |

## 8. Turning it off

| Goal | Do this |
|---|---|
| Stop everything, now | Set `provider: none` (or remove the `autonomy:` block) and apply. Behaviour is then identical to before. |
| Keep measuring, never act alone | Set `mode: shadow` and apply. Graduated actions stop acting at the next call. |
| One action back to supervised | Click **Demote now** on the Autonomy screen, or `POST /platform/v1/autonomy/<action>/demote`. |
| Something went wrong | Click **Report a wrong call**: it is recorded and the action is demoted on the spot. |

## 9. Known issues and open questions

**Known issues**

- **The recipient (resolved 2026-10-04).** It is now looked up rather than guessed (§3).
  Live, a routine message to a listed customer passes all seven checks.
  - A tenant must keep its contact list current: a customer not on it gets a person's
    check.
  - A CRM lookup (for example HubSpot) is the natural third source, and is not built.
- **Model variance on routine messages.** In the live re-check, one routine how-to message
  (a password-reset tip) escalated because the sensitivity score's confidence was 0.82
  against 0.90, though the same message passed in an earlier run. It errs toward a person.
  Shadow traffic will show whether 0.90 is right for this tenant.
- **Latency against the deadline.** Live calls took 0.6–2.7 s (median 0.6–0.8 s). Calls
  slower than `timeout_seconds` escalate. Measure the slowest 5% in shadow mode before
  choosing a timeout for enforce mode.
- **What cannot be observed:**
  - approved-after-editing (the gate only approves or refuses);
  - who answered in chat (not in the payload);
  - a chat "deny" also covers an interrupted turn.

  The ledger and the screen say so.
- **State is per action, tenant-wide**, as the schema has it. The ledger also breaks down
  by agent. Per-agent graduation would be a schema change.
- **Not live-proven.** Nothing has run in a real worker under the dispatcher, and no tenant
  has graduated an action. The capability catalog says `tested`, not `live_proven`.

**Open questions for TypeSafe** (unchanged from the audit, Q5). We need answers on these
before `full_args` or `enforce` is offered to a customer:

1. Pricing per token or per question. The report computes cost only from prices you
   supply.
2. Rate limits and quotas per key.
3. Data retention: are `state` and `questions` stored, for how long, and are they used for
   training?
4. Processing region(s); EU or Norway processing; a DPA and the sub-processor list.
5. Model pinning: can a request name `jev-1.13.0`, and how long is a version served?
6. Latency: typical and p99, and any SLA.
7. Calibration guidance for nouls used as safety gates.

Confirmed live (2026-10-02): the documented request and response shapes,
`jev-latest` → `jev-1.13.0`, and token usage reported per call.

## 10. Operating it

| Want | Use |
|---|---|
| See state, record and recent verdicts | Control Centre → **Autonomy** (viewers read; admins act) |
| Confirm a promotion | Autonomy → **Approve promotion** (admin) |
| Run reviews on a schedule | `nova autonomy review <bundle>` |
| A report for a customer or an investor | `nova autonomy report <bundle> --out reports/ [--minutes-per-approval 3] [--price-input-per-mtok X --price-output-per-mtok Y]` |
| Approve a held call in task work | Work or Approvals → **Approve** / **Refuse** |

## 11. Evidence

| Claim | Proof |
|---|---|
| Provider client, deadline, refusal of undocumented responses | `tests/platform/test_autonomy_provider.py` |
| The hook: off is off, shadow never acts, enforce rules, fail-closed paths, redaction, audit content | `tests/platform/test_autonomy_triage.py` |
| Ledger counts, tenant isolation, chat and board joins | `tests/platform/test_autonomy_ledger.py` |
| Proposal, admin-only confirm, intent → committed, crash leaves an open intent, demotion before the next call, model change | `tests/platform/test_autonomy_promotion.py` |
| Report arithmetic, labels, no invented cost, the commands | `tests/platform/test_autonomy_report.py` |
| Approval in task work, the "Always" fix | `tests/platform/test_task_approvals.py` |
| Live provider calls, wording, latency, tokens | `AUTONOMY_AUDIT.md`, "Live check" sections |

Each safety rule was also checked by deliberately breaking it (ignoring shadow mode,
letting the fake provider through, ignoring a model change, dropping the tenant filter,
skipping the re-read, confirming without eligibility), and a test failed every time.

---

## Design notes: deliberately not built yet

**Plain-language rule authoring.** A business owner writes "never promise a delivery date"
and NOVA compiles it into a noul with a threshold. Sketch:

- The owner's sentence becomes a noul statement.
- The threshold starts strict (0.10) and is shown with live examples from shadow traffic.
- NOVA proposes the rule; it is not applied until an administrator accepts it.
- A changed rule resets that action's agreement window, because trust was earned on the
  old questions.

The risk to design against is a vague sentence that fails open. Every compiled rule must
be tested against a small set of positive and negative examples the owner writes, before
it can be used.

**Nightly drift evaluation.** Re-run a sample of recent transcripts (with consent and in
the tenant's data mode) through the current model and question set, and compare with the
recorded verdicts and outcomes. Rising disagreement would demote, like a model change,
before real traffic shows it. Open points:

- where the sample is stored, and how long it is kept;
- keeping the cost within one triage call per sampled message;
- running it from the scheduler NOVA already governs, rather than a new daemon.
