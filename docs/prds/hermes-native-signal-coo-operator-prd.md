# Hermes Native Signal COO Operator PRD

Status: Draft for scope reset
Date: 2026-06-24
Product name: Torben
Primary interface: Signal
Primary repo: `/Users/ericfreeman/.hermes/hermes-agent`
Related brief: `docs/build-briefs/hermes-native-signal-coo-operator.json`

## 1. Product Thesis

Build Torben: a single Hermes-native Signal operator that behaves like a COO for Eric's life and work. Torben is the only user-facing interface. It coordinates hidden sub-operators for EA, GTM, and Finance, keeps working memory across replies, and turns periodic automation from static reports into actionable, conversational operating loops.

The product standard is conversation, not reporting. Torben should speak with the useful context already loaded: who the meeting is with, what the prior conversation covered, why the trade or article matters now, what has been drafted, what action is staged, and what Eric needs to approve or correct.

The product should feel like talking to a more organized, structured, decisive version of Eric. It should be direct, opinionated, brief when possible, and open to correction. It should not feel like a cron report, chatbot help desk, or set of disconnected agents.

The reset intentionally moves OpenClaw out of the critical path. Existing OpenClaw, Floki, Magnus, Ratatosk, and GBrain work can be used as references only when a component is proven useful.

Model routing:

- OpenAI is the default model lane for Torben, EA, Finance, memory, policy checks, and orchestration.
- Magnus/GTM is the Grok lane. Use `xai-oauth` / `grok-4.3` for X-native search, social review, founder-voice tuning, and X/LinkedIn content workflows.
- Grok should not become the default for EA or Finance just because GTM needs X context.
- X publishing credentials remain separate from Grok search/review credentials and stay approval-gated.

## 2. Problem

The current system produces reports, but the reports do not become durable work. When Eric replies, the system often does not know what message, decision, source context, or action he is responding to. That creates extra work: Eric has to restate context, interpret stale cron output, and mentally coordinate multiple agents.

The system also over-relies on static deterministic crons. Static code is good for collection, validation, policy checks, and ledger writes. It is bad at deciding what matters today, when to interrupt, how to phrase a useful recommendation, and how to continue a conversation after Eric responds.

The new product must solve two failures at once:

1. Preserve conversation and action context across every Signal exchange.
2. Let LLM-driven workflows reason under explicit constraints instead of encoding judgment into static scripts.

## 3. Goals

- One Signal thread for Eric.
- One named operator persona: Torben.
- Hidden sub-operators for EA, GTM, and Finance.
- OpenAI-backed default reasoning for Torben/EA/Finance.
- Grok-backed Magnus/GTM routing for X-native research and social content review.
- Every actionable outbound message has an action handle.
- Every reply resolves to the original action, source evidence, scope, allowed next actions, and current status.
- EA is the first executable slice.
- GTM drafts complete posts for approval and learns from engagement.
- Finance supports hard-limited live trading with about 1000 USD initial capital, first-iteration options and margin requirements, and separate personal finance analysis through Monarch.
- GBrain is optional and must earn its place through a bakeoff against local memory.
- Legacy OpenClaw/Hermes host cleanup is part of the reset prerequisite work. The old bridge should not stay resident and consuming compute unless a service is explicitly allowlisted for Torben.

## 4. Non-Goals

- No OpenClaw bridge repair or continuation as the product architecture. Decommissioning and host cleanup are in scope as reset hygiene.
- No separate user-facing agent channels.
- No unbounded external mutations. EA mutations are allowed only through explicit approval, pre-authorized policy, promotion gates, action ledger entries, and provider verification.
- No automatic GTM publishing in the first 30 days.
- No unlimited trading.
- No transfers, withdrawals, credential mutation, or account-link mutation in v0.
- No assumption that broker-specific margin, futures, or options execution is available until the connected platform confirms eligibility and tool support.
- No GBrain dependency before local action memory works.
- No direct reuse of Floki, Magnus, or Ratatosk modules unless the component is proven and explicitly accepted.

## 5. Target User Experience

Eric communicates in one Signal thread.

Torben sends compact, action-first messages:

```text
Torben / EA / Daily Brief / 2026-06-24

Decision: protect 10:30-12:00 for proposal work.
Why: two calendar conflicts and one email thread make the afternoon fragile.

Recommended:
[EA-20260624-001] Move 2:00 check-in to tomorrow.
[EA-20260624-002] Draft reply to Morgan with the revised timeline.
[EA-20260624-003] Remind Dad after 5:00.

Reply with the handle or say what to change.
```

Eric can reply:

```text
EA-20260624-002 make it shorter and less apologetic
```

Torben must resolve that reply to the original email thread, the prior recommendation, the relevant evidence bundle, and the allowed action. It should not ask Eric to restate who Morgan is or what thread this is.

If the reply is ambiguous:

```text
I have two open Morgan actions: timeline reply and invoice follow-up. Which one do you mean?
```

If the reply is stale or unsafe:

```text
That action expired because the source email changed. I can re-check the thread and draft a new version.
```

Required conversational examples:

```text
You have a call with Kim from U&I in 5 minutes.

Goal: close funding for your startup.
Last conversation: she pushed on competitive landscape and wants clarity on how you attack the market and the buyer.

Recommended line: lead with the buyer wedge, then explain why the competitive landscape is noise until buyers feel the pain weekly.
```

```text
There are strong new arXiv papers on agents and MCP that we can turn into a thought piece about observability and scoring agent systems.

I drafted the article and image direction.

Thesis: "As agents scale, so does the need to see what happens."
Angle: most teams are still measuring demos, not operational reliability. The piece argues for scorecards across trace quality, tool failure recovery, policy compliance, and human handoff quality.

[GTM-20260624-004] Review draft and image set.
```

```text
There is a scheduled hurricane path that could disrupt Gulf energy infrastructure while Strait shipping risk is already elevated.

Trade idea: a 2-day oil volatility position expressed through supported options or a futures-capable broker path.

I checked the current account, risk policy, and available instruments. The staged version is this many contracts, this premium, this max loss, this breakeven, this exit rule, and this expected payoff if the thesis hits.

[FIN-20260624-002] Approve staged trade review or ask for a smaller risk version.
```

## 6. Operator Persona

Name: Torben.

Voice:

- Direct.
- Decisive.
- Structured.
- Sparse unless detail matters.
- Comfortable pushing back.
- Open to correction.
- No corporate softness.
- No fake certainty.
- No long summaries when a decision is needed.

Torben should prefer:

- "Do this now because..."
- "I would not do this yet because..."
- "This is stale; I need to re-check it."
- "Blocked until you confirm X."
- "I drafted it. It is staged, not sent."

Torben should avoid:

- Generic assistant language.
- Repeating raw source material.
- Asking for context that is already in memory.
- Reporting activity without a recommendation.
- Saying an action happened unless a verified executor recorded it.

## 7. Operating Model

Crons are wake-up triggers, not decision makers.

Static code owns:

- Source collection.
- Data normalization.
- Secret redaction.
- Evidence IDs.
- Policy gates.
- Action handle generation.
- Action ledger writes.
- External executor calls.
- Provider response verification.

LLMs own:

- Prioritization.
- Interpretation.
- Drafting.
- Recommendation.
- Interruption judgment.
- Follow-up conversation.
- Voice and synthesis.

Every LLM output must be structured into a decision envelope before it can be shown to Eric or used by an executor.

External mutations are provider-side or outside-world writes. Examples:

- Sending an email, text, DM, reply, or public post.
- Archiving, labeling, deleting, or marking email.
- Creating, editing, accepting, declining, or deleting calendar events.
- Creating, completing, or changing reminders and tasks.
- Updating contacts, family follow-up records, or external CRM-like records.
- Creating, editing, categorizing, or marking financial transactions in Monarch.
- Placing, canceling, or modifying broker orders.
- Moving money, linking accounts, or changing credentials.

Mutation policy:

- Drafting, summarizing, organizing, and local open-loop tracking are free.
- External mutations require a validated decision envelope.
- Low-risk repeated mutations can be pre-authorized only with a narrow action, hard cap, provider/account pointer, and immediate report-back.
- High-risk mutations always require explicit approval in the Signal thread.
- Every mutation must write an action-ledger record before execution and a provider-verification record after execution.
- If provider verification is missing, Torben must say the action is unverified, not done.

## 8. System Shape

```text
Signal thread
  -> Hermes Signal adapter
  -> Torben operator
     -> inbound authorizer
     -> action ledger
     -> conversation memory
     -> scope router
        -> EA sub-operator
        -> GTM sub-operator
        -> Finance sub-operator
     -> LLM decision envelope
     -> policy gate
     -> staged action or external executor
     -> Signal response
```

Sub-operators are implementation details. Eric should not need to talk to Floki, Magnus, Ratatosk, or separate threads.

## 9. Core Data Concepts

### 9.1 Action Handle

Every actionable outbound item must get a unique handle.

Format:

```text
<SCOPE>-<YYYYMMDD>-<NNN>
```

Examples:

- `EA-20260624-001`
- `GTM-20260624-002`
- `FIN-20260624-003`

Each handle stores:

- handle
- scope
- outbound message ID
- source evidence IDs
- created time
- expiry time
- current status
- allowed next actions
- risk class
- user-visible summary
- executor state
- resolution history

### 9.2 Evidence Bundle

Evidence bundles are the bounded source inputs for a decision.

Examples:

- Email thread metadata and redacted excerpts.
- Calendar event metadata.
- Reminder state.
- Social post draft and engagement metrics.
- Market data snapshot.
- Broker position snapshot.
- Monarch transaction or budget summary.

Evidence bundles must have stable IDs. LLM prompts should receive evidence IDs and bounded content, not unbounded raw inboxes, account dumps, or browser state.

### 9.3 Decision Envelope

Every LLM recommendation must compile to a structured envelope:

```json
{
  "scope": "ea",
  "intent": "draft_email_reply",
  "summary": "Draft a shorter reply to Morgan about the revised timeline.",
  "evidence_ids": ["email-thread:abc123"],
  "recommended_action": "stage_draft",
  "requires_approval": true,
  "risk_level": "medium",
  "allowed_next_actions": ["revise", "approve_send", "discard"],
  "blocked_actions": ["send_without_approval"],
  "user_message": "I drafted a shorter version. It is staged, not sent."
}
```

Executors can only act on validated envelopes.

## 10. Memory Requirements

Local memory is required for v0.

Memory layers:

1. Action ledger: authoritative record for handles, state, replies, and executions.
2. Conversation memory: compact state for preferences, active topics, and recent decisions.
3. Evidence store: source-bound artifacts with stable IDs.
4. Optional GBrain: retrieval assist only after bakeoff.

The action ledger is always authoritative. GBrain can help recall related context, but it cannot replace action handles or executor state.

Minimum memory behavior:

- Resolve direct handle replies.
- Resolve obvious natural-language replies to recent handles.
- Refuse ambiguous replies and ask a narrow clarification.
- Refuse expired replies and offer to refresh.
- Preserve user corrections as future preferences when appropriate.
- Keep per-scope state separate while preserving one user-facing thread.

## 11. Autonomy Model

Use four levels:

- A: Read and report.
- B: Draft.
- C: Stage for approval.
- D: Execute within policy.

Initial policy:

- EA: B/C by default for first 30 days, with D allowed for explicitly approved single actions and narrow pre-authorized recurring actions after the approval workflow exists.
- GTM: B/C for first 30 days.
- Finance trading: D allowed only after hard limits, kill switch, broker audit path, and instrument-specific risk checks exist. Options and margin are in first-iteration scope.
- Monarch personal finance: A/B/C initially, with C/D for transaction categorization or budget changes only after explicit policy and approval.
- GBrain: no autonomy. Evaluation-only until proven.

Promotion requires evidence. No scope promotes because it "seems to work."

## 12. EA Scope

EA is the first executable slice.

EA must use core parts of Juno from:

```text
/Users/ericfreeman/.openclaw/workspace/floki-agent/downloads/juno-home-chief-of-staff-v1.0.0-20260619
```

Juno components to reuse as product requirements:

- Five-step loop: notice, organize, prepare, recommend, wait for approval.
- Daily brief pattern: short morning handoff, signal over inventory, one recommended next action.
- Inbox and calendar triage: sort inbound into info, reply, deadline, print, or flagged; draft responses and calendar fixes.
- Open-loop tracker: every loose end has exactly one state: next-action, waiting-on, deferred-until, dropped, or done.
- Approval rules: external actions require explicit approval unless a narrow pre-authorization exists.
- Decision packet: options, upside, downside, cost, risk, recommendation, approval request, and blocked actions.

Torben can be more decisive and less cozy than Juno, but the EA safety spine should come from Juno.

Responsibilities:

- Daily brief.
- Email triage.
- Calendar prep.
- Meeting and call prep.
- Reminder and family follow-up.
- Action memory.
- Operational day structure.
- ADHD support through prioritization and interruption filtering.

EA should answer:

- What matters today?
- What can be ignored?
- What needs Eric's decision?
- What can Torben draft or stage?
- What can Torben execute now because Eric already approved or pre-authorized it?
- What is at risk of being dropped?
- What changed since the last brief?

EA must not:

- Send email without approval or a narrow pre-authorization.
- Archive, delete, label, or mark email without approval or a narrow pre-authorization. Deletion always requires explicit approval.
- Create, edit, accept, decline, or delete calendar events without approval or a narrow pre-authorization.
- Pretend an action is done when it is only drafted.

EA mutation ladder:

1. Draft: produce the message, reminder, calendar edit, or task locally.
2. Stage: attach evidence, handle, destination, and exact mutation preview.
3. Approve: Eric approves by handle or a policy pre-authorizes the exact action class.
4. Execute: provider adapter performs the write.
5. Verify: provider response and read-after-write prove the mutation happened.
6. Report back: Torben says what changed, what failed, and what remains open.

EA pre-authorizations must follow Juno's four-part pattern:

- The action is narrow and specific.
- The cap is explicit, including time, money, scope, or recipient limit.
- The account/provider pointer is explicit.
- The report-back is immediate.

Example approved EA mutation:

```text
[EA-20260624-011] Move 2:00 check-in to tomorrow at 11:30.

Approval means: update the Google Calendar event, keep attendees unchanged, and report back with the new event time.
```

Daily brief shape:

```text
Torben / EA / Daily Brief

Top decision:
Recommended day structure:
Needs Eric:
Already staged:
Watch:
Family/personal:
Done since last brief:
```

EA success metrics:

- Eric replies with less context restatement.
- Staged actions are approved or revised, not ignored.
- Daily brief leads with decisions, not inventory.
- Fewer missed replies and personal follow-ups.
- Calendar conflicts are surfaced before they matter.

## 13. GTM Scope

GTM turns Eric's public profile into a compounding thought leadership and GTM engine.

Responsibilities:

- Research market, security, startup, AI, and category themes.
- Draft complete posts for LinkedIn and X.
- Draft article-length thought pieces and derivative social posts.
- Generate or stage image directions/assets to accompany approved content.
- Adapt Eric's founder voice plus Magnus-style sharpness.
- Track post performance.
- Learn from engagement over time.
- Suggest follow-up posts, replies, and threads.
- Connect public profile work to Eric's businesses.

Voice target:

- Sharp.
- Crass when useful.
- Direct.
- High-conviction.
- Memorable.
- Not generic brand-safe content.
- Never a personal dunk just for engagement.

GTM must not:

- Publish automatically in first 30 days.
- Reply publicly without approval.
- Like, repost, DM, or follow without approval.
- Optimize purely for vanity metrics.

GTM post envelope:

```text
Torben / GTM / Draft

Thesis:
Audience:
Why now:
Sources:
Draft:
Image direction:
Risk:
Engagement hypothesis:
Approval handle:
```

GTM example output:

```text
There are strong new arXiv papers on agents and MCP that can become a piece on observability.

Recommendation: publish a sharp LinkedIn article, then split it into an X thread and two replies.

Working title: "Your agent demo is lying to you."
Core line: "As agents scale, so does the need to see what happens."
Asset: generated diagram showing agent traces, tool calls, policy gates, and human handoff scoring.

[GTM-20260624-007] Review the draft and image direction.
```

GTM success metrics:

- Draft acceptance rate.
- Posts published per week.
- Engagement quality, not just raw impressions.
- Follower growth in the right audience.
- Inbound business relevance.
- Reuse of winning voice patterns.

## 14. Finance Scope

Finance has two arms:

1. Trading and investment operations.
2. Personal finance operations.

Trading target:

- Robinhood Agentic MCP first.
- About 1000 USD starting capital.
- Live trading is allowed only after hard controls exist.
- Paper trading is not the goal because it did not produce useful signal for Eric.
- Options are in first-iteration scope.
- Margin is in first-iteration scope if the account, broker, and agentic tool path confirm eligibility.
- Event-driven macro trades are in scope, including weather, geopolitical, supply-chain, energy, rates, and volatility theses.
- Futures-style ideas are in scope for research and staging. Execution requires a supported broker/tool path; if Robinhood Agentic does not expose futures execution, Torben must propose supported proxies such as equities, ETFs, or options, or recommend adding a futures-capable integration.

Personal finance target:

- Monarch Money MCP for account, transaction, budget, cashflow, and spending analysis.
- Monarch can produce recommendations and staged actions.
- Monarch can stage transaction categorization, budget, savings, recurring-subscription, and expense-cutting actions.
- Monarch mutations require explicit policy and approval. The PRD should not treat Monarch as read-only forever.

External reference constraints:

- Robinhood describes Agentic Trading as MCP-based access for AI agents, with a dedicated funded account, visible activity, trade notifications, and disconnect controls. It also discloses significant risk, including possible loss of the entire investment and responsibility for monitoring the agent.
- Robinhood's current support page says the agent can place long equities and options orders, and that options trading is rolling out and may not be available to everyone.
- Robinhood's current Agentic Trading docs list account/portfolio, watchlist, market-data, equities, and options tools, including order review and order placement tools.
- Robinhood margin docs say margin access is not automatic and requires eligibility; margin involves interest, maintenance requirements, possible margin calls, and potential liquidation without prior approval.
- The Monarch MCP listing describes access to accounts, transactions, budgets, cashflow, analytics, MFA-supported authentication, persistent sessions, and transaction management.

Finance hard limits required before any live trade:

- Total capital budget.
- Per-position cap.
- Per-order cap.
- Daily loss cap.
- Weekly loss cap.
- Max open positions.
- Max margin used.
- Max options premium.
- Max contracts per trade.
- Max days to expiration or minimum days to expiration, depending on strategy.
- Max spread width for defined-risk spreads.
- Expiration and assignment-risk policy.
- Allowed asset list.
- Disallowed asset list.
- Allowed strategy list.
- Disallowed strategy list.
- Market-hours policy.
- Cooldown after loss.
- Kill switch.
- Manual override.
- Broker response audit.
- Read-after-write reconciliation.

Default v0 finance policy unless Eric overrides:

- Total capital: 1000 USD.
- Max single position: 10 percent of finance capital.
- Max daily realized loss: 3 percent of finance capital.
- Max weekly realized loss: 7 percent of finance capital.
- Max open positions: 3.
- Options: enabled for first iteration after account/tool eligibility is confirmed.
- Options strategy default: long calls, long puts, and defined-risk debit spreads only.
- Options default block: naked short options, uncovered calls, undefined-risk spreads, same-day expiration, and any strategy whose max loss cannot be computed before order review.
- Options max premium: 5 percent of finance capital per trade until Eric changes policy.
- Options max contracts: 1 contract per trade until Eric changes policy.
- Margin: in scope for first iteration after account/tool eligibility is confirmed.
- Margin borrow cap: min(250 USD, 25 percent of finance capital) until Eric changes policy.
- Margin block: no trade that can trigger unresolved maintenance risk under the current account snapshot.
- Futures: research and staging allowed; live futures execution blocked unless a supported broker/tool path is added and a futures-specific risk policy exists.
- No transfers.
- No withdrawals.
- No credential or account-link mutation.

Finance trade envelope:

```json
{
  "scope": "finance",
  "intent": "place_trade",
  "thesis": "...",
  "asset": "...",
  "instrument_type": "equity_or_option_or_supported_future_proxy",
  "side": "buy",
  "order_type": "market_or_limit",
  "notional_usd": 75,
  "max_loss_usd": 75,
  "max_profit_usd": null,
  "breakeven": null,
  "expiration": null,
  "contracts": null,
  "margin_used_usd": 0,
  "exit_rule": "...",
  "risk_checks": {
    "capital_budget": "pass",
    "position_cap": "pass",
    "daily_loss_cap": "pass",
    "asset_allowlist": "pass",
    "strategy_allowlist": "pass",
    "options_max_loss_known": "pass",
    "margin_cap": "pass"
  },
  "kill_switch_state": "off",
  "requires_user_notice": true
}
```

Finance must block execution if any risk check fails. The explanation to Eric must name the exact failed rule.

Finance conversational example:

```text
There is a scheduled hurricane path that could disrupt Gulf Coast production while shipping risk is already elevated.

Recommendation: do not buy oil headlines blindly. Stage a 2-day volatility thesis and express it only through an instrument with known max loss.

Candidate: long put/call structure or defined-risk spread on the supported oil proxy with liquidity above threshold.
Cost: $X premium.
Max loss: $X.
Breakeven: $Y.
Target exit: close at +Z percent or before expiration risk crosses policy.
Blockers: margin eligibility, options availability, spread width, and broker review warnings.

[FIN-20260624-004] Review the staged trade. It is not placed yet.
```

Finance success metrics:

- Risk policy never violated.
- Every order has a complete audit trail.
- Realized and unrealized P/L tracked.
- Theses are reviewed after outcomes.
- Losing strategies are stopped, not rationalized.
- Monarch recommendations identify real savings or cashflow improvements.

## 15. Signal Interaction Requirements

Inbound:

- Accept only configured Signal identities.
- Record inbound message ID, sender, timestamp, raw text hash, and parsed intent.
- Resolve handles before invoking a sub-operator.
- If no handle exists, route by recent active context or ask a narrow clarification.

Outbound:

- Every actionable item gets a handle.
- Every message says whether an action is drafted, staged, executed, blocked, or expired.
- No naked yes/no prompts.
- No raw dumps unless Eric asks.
- Include enough context to act without forcing him to open another system.

Message priority:

- Urgent interruption.
- Daily operating brief.
- Approval needed.
- FYI / low-priority digest.
- Follow-up after user reply.

Interruption policy:

Interrupt immediately only for:

- Same-day calendar risk.
- Email requiring urgent user decision.
- Family/personal follow-up that is time-sensitive.
- Finance kill-switch event or risk breach.
- Live broker execution or failure.
- Security-sensitive account issue.

Batch everything else.

## 16. LLM Constraints

LLMs must:

- Use only supplied evidence.
- State uncertainty.
- Produce structured envelopes.
- Respect scope-specific autonomy.
- Prefer a recommendation over a summary.
- Ask one narrow question when blocked.
- Preserve Eric's voice in drafts.

LLMs must not:

- Invent source context.
- Claim external execution.
- Bypass policy gates.
- Use stale evidence as fresh.
- Turn every cron into an interruption.
- Expose credentials, tokens, or raw financial data.

## 17. Security and Privacy Requirements

Sensitive surfaces:

- Signal identity.
- Gmail.
- Google Calendar.
- LinkedIn.
- X.
- Robinhood.
- Monarch.
- OpenAI Codex OAuth for default Torben/EA/Finance reasoning, with API key fallback only if OAuth is not viable.
- xAI/Grok auth for Magnus/GTM X-native research and review.
- Possible Gemini or other exchange credentials later.

Required controls:

- No credentials committed to repo.
- OAuth/provider-native auth and registered MCP sessions are the primary runtime auth plane.
- 1Password is optional bootstrap for non-OAuth static values only; repo-owned config may contain `op://` secret references, never resolved secret values.
- Long-running services should start through the direct Torben wrapper by default; use the `op` wrapper only for integrations that still require static secret injection.
- If the optional 1Password path is used under launchd, service accounts should be least-privilege and vault-scoped. The service account bootstrap token must live outside the repo, preferably in macOS Keychain or an explicitly managed launchd secret store.
- If the optional runtime uses interactive 1Password app auth instead of a service account, the `op` wrapper must fail closed after reboot or lock until the operator unlocks 1Password.
- Least-privilege credentials where available.
- Separate account or budget boundary for agentic trading where possible.
- Redacted logs.
- Evidence IDs over raw payloads in prompts where possible.
- Audit trail for every external mutation.
- Explicit fail-closed behavior on missing policy.
- Kill switch for finance.
- Prompt/log boundaries for Monarch data.

### 17.1 Runtime Auth Model

Torben must use OAuth/provider-native auth and registered MCP sessions first. 1Password is optional bootstrap for the remaining non-OAuth static values.

Primary patterns:

- OpenAI Codex OAuth through Hermes auth for Torben/EA/Finance orchestration.
- xAI/Grok OAuth through Magnus for GTM/X-native research and social review.
- Google OAuth for Gmail and Calendar.
- Registered MCPs for finance, including Robinhood Agentic MCP and Monarch Money MCP.
- Local `signal-cli` or gateway config for Signal, which is not OAuth.

Optional 1Password patterns:

- `op run --env-file <template-with-op-refs> -- <command>` for process-level environment injection.
- `op run --environment <environment-id> -- <command>` if 1Password Environments are used for the Torben runtime.
- `op read op://vault/item/field` for one-off secrets inside wrapper scripts.
- `op inject --in-file <template> --out-file <tmpfile>` only when an integration requires a config file; the resolved file must be written to a private temp directory, never the repo.

Disallowed patterns:

- Provider tokens in repo files.
- Provider tokens in LaunchAgent plists.
- Provider tokens in checked-in examples.
- Long-lived resolved `.env` files for Torben.
- Falling back from missing 1Password values to stale plaintext profile secrets.
- Making 1Password service-account setup a global blocker for OAuth/MCP-native runtime checks.

Required auth registrations:

- OpenAI Codex OAuth for the default Torben lane.
- xAI/Grok OAuth for Magnus/GTM.
- Signal account, daemon URL, and any Signal gateway token when Signal is enabled.
- Gmail and Google Calendar OAuth registrations.
- LinkedIn and X API or browser automation credentials.
- Robinhood Agentic MCP registration for live trading.
- Monarch MCP registration for personal finance.
- Any future Gemini/exchange credentials.

Legacy `.env` files can exist only as migration inputs. Before Torben live launch, each required value must be either replaced by OAuth/MCP-native auth, moved to optional 1Password bootstrap, explicitly rejected as obsolete, or quarantined outside the runtime path.

### 17.2 Host Cleanup And Decommissioning

The reset should stop compute spend from obsolete services.

Current read-only host inventory found likely cleanup candidates:

- `ai.openclaw.gateway`
- `ai.openclaw.cron-receiver`
- `ai.openclaw.filter`
- OpenClaw-managed `signal-cli` daemon
- `ai.hermes.gateway-floki`
- `ai.hermes.gateway-magnus`
- `ai.hermes.gateway-ratatosk`
- `ai.gbrain.mcp`
- `com.hermes.keepawake`
- `com.hermes.operational-health`
- `com.hermes.auto-update`
- `com.magnus.x-oauth2-keepalive`
- Ratatosk launchd jobs such as crypto shadow, Gemini reconcile/exit/policy, Robinhood refresh, daily Signal workflow, and circuit breaker.

Cleanup policy:

- Inventory first, stop second, delete/quarantine last.
- Generate an allowlist before stopping anything.
- Stop launchd jobs with `launchctl bootout` or `launchctl disable`, not ad hoc process killing, unless a child process survives after the service is unloaded.
- Preserve state, logs, and plists in a timestamped quarantine directory before removing runtime hooks.
- Do not remove source repos as part of compute cleanup.
- Do not remove 1Password, Signal Desktop, or user-level apps unless explicitly requested.
- After cleanup, prove no old OpenClaw/Hermes/Ratatosk/Magnus/Floki services are loaded except the allowlist.

## 18. Observability

The system must be able to answer:

- What did Torben send?
- Why did Torben send it?
- What evidence did it use?
- What handle did Eric reply to?
- What scope owned the action?
- What did the LLM recommend?
- What did policy allow or block?
- What was staged?
- What was executed?
- What provider response verified execution?
- What changed after execution?

Required artifacts:

- Action ledger.
- Conversation memory compaction.
- Evidence bundles.
- Decision envelopes.
- Policy decisions.
- Executor audit records.
- Finance P/L and risk reports.
- GTM engagement reports.
- EA daily brief history.

## 19. Rollout Plan

Phase 0: PRD and scope approval.

Phase 0a: Runtime security and host cleanup.

- Create Torben 1Password vault/item map.
- Replace Torben runtime secrets with `op://` references or 1Password Environment variables.
- Add a launch wrapper that resolves 1Password secrets at process start and then execs Torben.
- Inventory legacy launchd jobs and resident processes.
- Stop or disable obsolete OpenClaw, Hermes profile gateways, GBrain, and legacy Ratatosk/Magnus/Floki jobs unless explicitly allowlisted.
- Quarantine old plists/env files before deletion.
- Prove no obsolete services remain loaded.

Phase 1: Torben memory core.

- Implement local action ledger.
- Implement Signal reply resolver.
- Prove handle resolution with tests.
- No external mutations in this phase.

Phase 2: EA first slice.

- Generate daily brief from fixture evidence.
- Stage email/calendar/reminder actions.
- Preserve reply context.
- Run one live Signal canary with no external mutations.
- Use Juno daily-brief, inbox-triage, open-loop-tracker, decision-packet, and approval-rule patterns.

Phase 3: EA approval workflow.

- Allow Eric to approve staged drafts.
- Executors verify provider response before marking complete.
- Keep mutation count explicit.
- Add approved and pre-authorized EA mutation support for email, calendar, reminders, tasks, and labels.
- Keep deletion, money, legal, tax, health, and irreversible actions explicit-approval only.

Phase 4: GTM draft workflow.

- Research and draft complete posts.
- Add voice rubric.
- Add engagement ingestion.
- Keep publishing approval-gated.

Phase 5: Finance risk core.

- Implement finance policy contract.
- Add mock broker, mock options chain, mock margin-account, and mock Monarch fixtures.
- Add kill switch and audit trail.
- Add equity, options, and margin risk gates before connecting Robinhood Agentic MCP.
- Only then connect Robinhood Agentic MCP.

Phase 6: Monarch personal finance.

- Read accounts, budgets, transactions, and cashflow summaries.
- Produce staged savings and expense recommendations.
- Add approval-gated transaction/category/budget mutations only after policy and redaction tests pass.

Phase 7: GBrain bakeoff.

- Compare local-only memory versus local plus GBrain.
- Keep GBrain only if correctness or usefulness improves without too much noise or latency.

## 20. Validation Plan

Required tests before live canary:

- OAuth/MCP-native auth policy validates with `onepassword_bootstrap=optional`.
- Optional 1Password CLI bootstrap validates only when a selected integration still needs static secret injection.
- Torben `op` wrapper fails closed when selected and a required `op://` value is missing.
- Torben launch wrapper does not write resolved secrets into the repo.
- LaunchAgent plist contains no provider tokens.
- Legacy host cleanup inventory has been reviewed and allowlisted.
- Legacy services are stopped/disabled except the allowlist.
- Signal inbound authorization.
- Action handle generation.
- Action handle expiry.
- Direct handle reply resolution.
- Ambiguous reply clarification.
- Expired reply safe refusal.
- Scope router selection.
- Decision envelope validation.
- EA daily brief fixture.
- EA staged action fixture.

Required tests before GTM publishing:

- Draft envelope validation.
- Voice rubric.
- No publish without approval.
- Engagement ingestion with fixture data.
- Browser/API failure handling.

Required tests before finance live trading:

- Risk policy fail-closed behavior.
- Per-order cap block.
- Position cap block.
- Daily loss cap block.
- Weekly loss cap block.
- Asset allowlist block.
- Strategy allowlist block.
- Options max-loss calculation.
- Options unsupported-account block.
- Margin eligibility block.
- Margin cap block.
- Futures unsupported-tool block.
- Kill switch block.
- Broker call audit.
- Broker failure handling.
- Post-order reconciliation.

Required tests before Monarch live data:

- Prompt redaction.
- Log redaction.
- Read-only summaries.
- Transaction/action staging.
- Approved transaction/category mutation with provider verification.
- MFA/session failure handling.

## 21. Definition of Done

The product is done for v0 when:

- Eric has one Signal thread with Torben.
- Torben sends an EA daily brief with action handles.
- Eric can reply to an action handle and Torben continues with the correct source context.
- Unknown, ambiguous, and expired replies fail safely.
- EA can draft, stage, and execute approved or pre-authorized actions with provider verification.
- GTM can research and draft complete posts with image direction/assets for approval.
- Finance scopes are defined with live-trading policy gates; live order paths remain disabled until risk tests and broker audit pass.
- Finance risk policy exists before any live order path.
- Monarch is scoped as personal finance insight plus approved staged mutations, not uncontrolled account mutation.
- GBrain is out of the critical path unless a bakeoff proves value.
- Torben runtime auth uses OAuth/provider-native registrations and registered MCP sessions first; optional 1Password bootstrap values are not stored in repo, plists, or long-lived `.env` files.
- Legacy OpenClaw/Hermes bridge services are stopped or explicitly allowlisted.
- The system has test coverage for the reply-memory behavior that caused the reset.

## 22. Open Decisions

These must be resolved before implementation or first live canary:

- Confirm final operator name: Torben or Heimdall.
- Confirm any remaining non-OAuth static secrets that need optional 1Password bootstrap.
- Confirm optional service-account bootstrap policy only if a selected static-secret integration requires it.
- Confirm old-service allowlist before stopping legacy OpenClaw/Hermes/GBrain/Ratatosk/Magnus/Floki launchd jobs.
- Confirm Signal account and thread.
- Confirm which Gmail accounts are in EA scope.
- Confirm which calendars are in EA scope.
- Confirm which family/personal contacts need follow-up tracking.
- Confirm the exact daily brief time and quiet hours.
- Confirm GTM source areas and prohibited topics.
- Confirm LinkedIn/X credential and browser automation strategy.
- Confirm finance capital amount if different from 1000 USD.
- Confirm finance allowed asset classes.
- Confirm finance daily and weekly loss caps.
- Confirm margin borrow cap and whether account value/eligibility supports margin.
- Confirm allowed options strategies and options approval level.
- Confirm whether futures should require a second broker/tool path or be expressed through supported proxies in v0.
- Confirm which Monarch mutations are allowed after approval: transaction category edits, budget changes, savings actions, recurring-subscription updates, or none.
- Confirm whether Gemini is deferred or re-added.
- Confirm GBrain bakeoff criteria.

## 23. Source Notes

- Robinhood Agentic Trading product page: https://robinhood.com/us/en/agentic-trading/
- Robinhood Trading with your agent support page: https://robinhood.com/us/en/support/articles/trading-with-your-agent/
- Robinhood Agentic Trading overview: https://robinhood.com/us/en/support/articles/agentic-trading-overview/
- Robinhood margin overview: https://robinhood.com/us/en/support/articles/margin-overview/
- Robinhood minimum margin note: https://robinhood.com/us/en/support/articles/minimum-margin/
- Robinhood day trading / intraday margin note: https://robinhood.com/us/en/support/articles/day-trading/
- Monarch Money MCP listing: https://mcpmarket.com/server/monarch-money
- Monarch Money MCP GitHub reference: https://github.com/robcerda/monarch-mcp-server
- 1Password load secrets into environment: https://www.1password.dev/cli/secrets-environment-variables
- 1Password load secrets into scripts: https://www.1password.dev/cli/secrets-scripts
- 1Password service accounts with CLI: https://www.1password.dev/service-accounts/use-with-1password-cli

## 24. Repo Reference Map

The reset should reuse proven ideas and tests, not inherit the old OpenClaw/Hermes integration shape.

### 24.1 Interface, Signal, And Memory

Primary repo:

```text
/Users/ericfreeman/.hermes/hermes-agent
```

Use:

- `gateway/platforms/signal.py` as the Signal transport reference.
- `tests/gateway/test_signal.py`, `tests/gateway/test_signal_format.py`, and `tests/gateway/test_signal_rate_limit.py` as adapter safety references.
- `tests/gateway/test_reply_to_injection.py` as proof that reply pointers can be injected, while recognizing that Torben still needs a durable action ledger.
- `hermes_cli/subcommands/cron.py` as the reference for scheduled wake-up triggers.

Do not use:

- The current OpenClaw bridge as the product architecture.
- Raw cron reports as the user-facing experience.

Current clean profile:

- `/Users/ericfreeman/.hermes/profiles/torben/config.yaml`
- Default model: `openai-codex` / `gpt-5.5`
- GTM route: `magnus` profile using `xai-oauth` / `grok-4.3`
- Runtime secret template: `/Users/ericfreeman/.hermes/profiles/torben/runtime.env.op`
- Direct runtime wrapper: `/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes`
- Optional 1Password wrapper: `/Users/ericfreeman/.hermes/profiles/torben/bin/torben-op-hermes`
- Finance MCP registration plan: `/Users/ericfreeman/.hermes/profiles/torben/mcp-registration-plan.json`

### 24.2 EA / Chief Of Staff

Required reference:

```text
/Users/ericfreeman/.openclaw/workspace/floki-agent/downloads/juno-home-chief-of-staff-v1.0.0-20260619
```

Use:

- `README.md` and `persona/IDENTITY.md` for the five-step operator loop.
- `skills/daily-brief/SKILL.md` for short morning handoff requirements.
- `skills/inbox-triage/SKILL.md` for email/calendar triage buckets and safety boundaries.
- `skills/open-loop-tracker/SKILL.md` for open-loop states and aging.
- `templates/approval-rules.md` for explicit approval and pre-authorization semantics.
- `templates/decision-packet.md` for high-stakes decision packets.

Additional local reference:

```text
/Users/ericfreeman/ea-automation
```

Use:

- `src/onboarding/conversational.py` for conversational onboarding and preference capture.
- `src/mcp_server.py` for EA profile, calibration, VIP contacts, and mode thresholds.
- `skills/proactive/SKILL.md` for proactive meeting alerts, stale follow-ups, and day-structure checks.

### 24.3 GTM / Social / Thought Leadership

Primary reference:

```text
/Users/ericfreeman/magnus
```

Use:

- `src/magnus/article_recommendations.py` for source-backed article opportunity detection.
- `src/magnus/x_article_draft.py` for article draft packets, source notes, image prompts, and approval flow.
- `src/magnus/linkedin_content.py`, `src/magnus/platform_content.py`, and `src/magnus/content_strategy.py` for channel adaptation.
- `src/magnus/content_performance.py` and `src/magnus/content_feedback.py` for engagement learning.
- `src/magnus/voice_corpus.py` and `config/eric-voice-profile.md` for Eric's founder voice.
- `src/magnus/publishing_guard.py` for approval-gated public writes.
- `connectors/x_writer.py`, `connectors/x_reader.py`, and `connectors/postiz.py` as reference connector work, not automatic reuse.
- `scripts/openai_image_generate.py` as reference for generated supporting visuals.

Do not use:

- Any Magnus public write path until Torben approval handles and publishing guard checks are integrated.

### 24.4 Finance / Trading / Risk

Primary reference:

```text
/Users/ericfreeman/ratatosk
```

Use:

- `src/ratatosk/engine/risk_manager.py` for deterministic risk gates, capital caps, loss caps, concentration controls, and circuit breakers.
- `src/ratatosk/state/order_ledger.py` for durable idempotent order state before broker submission.
- `src/ratatosk/intelligence/options_analyzer.py` for options-chain analysis, defined-risk candidates, premium/max-loss framing, and advisory guardrails.
- `tests/test_engine/test_executor_robinhood.py` for supervised/live-trading guard tests.
- `docs/adlc-robinhood-equity-trading.md` as historical planning reference only.

Do not directly reuse:

- `src/ratatosk/connectors/robinhood.py` as the primary new connector, because the reset should target Robinhood Agentic MCP first. Keep it as historical reference for account/order concepts and tests.

Secondary finance reference:

```text
/Users/ericfreeman/prediction-market-analysis
```

Use:

- Prediction-market and event-analysis patterns for thesis calibration, event odds, and macro-disruption research.
- Do not treat it as a broker or execution path.

### 24.5 Personal Finance

External reference:

- Monarch Money MCP server and listing from Source Notes.

Use with Torben:

- Account, transaction, budget, cashflow, and net-worth analysis.
- Staged expense cuts, subscription review, budget changes, transaction categorization, and savings actions.
- Approval-gated mutations only after redaction and provider verification tests pass.

### 24.6 GBrain / Retrieval

Candidate references:

```text
/Users/ericfreeman/gbrain-fork
/Users/ericfreeman/brain
```

Use:

- `gbrain-fork` as a retrieval candidate for source-attributed recall and MCP surface patterns.
- `brain` as a historical knowledge corpus.

Do not use:

- GBrain as authoritative action memory.
- GBrain as a required dependency before local action ledger and evidence bundles work.

Bakeoff requirement:

- Same prompt set, local-only memory versus local plus GBrain.
- Measure correct handle resolution, useful recall, false context injection, latency, and operator usefulness.
- Keep GBrain only if it materially improves correctness or usefulness.

### 24.7 Planning / ADLC

Reference:

```text
/Users/ericfreeman/swelfare
```

Use:

- The latest remote ADLC schema and validator for build briefs and downstream implementation planning.
- This PRD remains the product source of truth; ADLC plans should compile from it, not replace it.

### 24.8 Torben Persona And Response Contracts

Reference:

```text
/Users/ericfreeman/torbins-workspace
```

Use:

- `torbin/contracts/message-response.md` for recommendation-first, concise, blocker-aware responses.
- `torbin/contracts/memory-capture.md` for durable memory capture after decisions and repeated failures.
- `torbin/decision-rules.md` for decisive operator behavior and explicit approval boundaries.
- `projects/agent-plans/AGENT_SCOPES_PLAN.md` as historical persona/scope reference only.
