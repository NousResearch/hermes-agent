---
name: gmail-inbox-audit
description: "Audit Gmail inboxes across multiple connected accounts via Google Workspace MCP: fetch metadata, categorise noise, identify filter targets, and write a triage report."
version: 1.0.0
author: KENSEI
metadata:
  hermes:
    tags: [Gmail, Google, Workspace, email, audit, triage, MCP]
    related_skills: [add-gmail-account, mailbox-agent]
adoption_status: provisional
---

# Gmail Inbox Audit via Google Workspace MCP

Analyse inbox clutter across multiple Gmail accounts connected through the Google Workspace MCP server. Identify top noise sources, estimate category breakdowns, compare accounts, and produce a triage report with concrete label/filter recommendations.

## When to Use

- You feel overwhelmed by unread email and want a data-driven cleanup plan.
- You want a periodic (monthly/quarterly) inbox health report across all Gmail accounts.
- You need to identify which senders or subject patterns are consuming the most attention.
- You want KENSEI-managed label recommendations with `kensei/` prefixes.
- You have multiple Gmail accounts (personal, secondary, business) connected via the Google Workspace MCP.

## Assumptions

- Gmail accounts are already authenticated via the Google Workspace MCP server (`google_workspace` MCP tools available).
- Default CLI sessions may run with `no_mcp` to avoid context bloat. If Gmail tools are not available, start a fresh mail-specific session with explicit toolsets, e.g. `hermes chat -t google_workspace,skills,memory,session_search -q "..."`.
- OAuth tokens are valid. If `start_google_auth` is needed, run the `add-gmail-account` skill first.
- Prefer `search_gmail_messages` for listing, then `get_gmail_messages_content_batch` with `format=metadata` for Subject/From/Date extraction.
- In Hermes Workspace, use **Operations** for a persistent mailbox digest agent and **Conductor** for one-off setup/audit/prompt generation. Do not use Swarm unless there are multiple independent mailbox lanes. If the user asks whether to paste prompts or have the agent execute, generate prompts internally and execute safe dry-run checks yourself, stopping before mailbox mutations or recurring schedules.

## Workflow

### 1. List accounts and pick targets

Use the accounts the user has already connected. Typical set:
- Primary personal
- Secondary / alias
- Studio / business

If unsure which accounts are connected, search session history or ask the user.

### 2. Fetch inbox listings

For each target account, run:

```python
# Pseudocode via MCP tool call
search_gmail_messages(query="in:inbox", page_size=100, user_google_email=ACCOUNT)
```

**Pagination:** If a `page_token` is returned, repeat with that token to exhaust the inbox (or stop at a representative sample, e.g. 200 messages per account).

**Rate-limit awareness:** The MCP server enforces Gmail API quotas. Do NOT fire multiple concurrent `get_gmail_messages_content_batch` calls against the same account. Serialise them with brief pauses, or batch-fetch with modest message counts (10-25 per call).

### 3. Estimate category breakdown via category searches

For each target account, run targeted category searches to estimate noise composition **before** fetching individual message metadata. This avoids expensive batch fetches on inboxes with 100+ messages.

```python
search_gmail_messages(query="in:inbox category:promotions", page_size=1, user_google_email=ACCOUNT)
search_gmail_messages(query="in:inbox category:social",     page_size=1, user_google_email=ACCOUNT)
search_gmail_messages(query="in:inbox category:updates",    page_size=1, user_google_email=ACCOUNT)
search_gmail_messages(query="in:inbox category:forums",     page_size=1, user_google_email=ACCOUNT)
search_gmail_messages(query="in:inbox category:purchases",  page_size=1, user_google_email=ACCOUNT)
```

Each response starts with the count (`Found N messages matching...`). Record these numbers per category per account.

**Tab-detection heuristic:** If the inbox is non-empty but ALL category searches return "No messages found", Gmail tabs exist as labels but the Inbox type is **not** set to "Default" (Tabs) in Settings → Inbox. All emails are landing in Primary instead. This is a key triage finding.

### 4. Fetch metadata in batches (revised guidance)

```python
get_gmail_messages_content_batch(
    message_ids=[ID1, ID2, ...],
    format="metadata",
    user_google_email=ACCOUNT
)
```

**Batch size guidance:** Start with 25 IDs. If you hit HttpError 429 ("Too many concurrent requests"), drop to 10 IDs per call and add a short delay. The metadata format avoids large body text, keeping responses compact.

**Extraction targets:**
- Subject
- From (sender domain and display name)
- Date
- List-Unsubscribe (presence indicates newsletter/subscription)

### 5. Categorise manually or with KENSEI

Group emails into buckets:
- Marketing / Promo / Newsletter
- Social / Community (Skool, Discord, etc.)
- Infrastructure / Monitoring (healthchecks, uptime alerts)
- Platform Admin (Apple, Google Play, Developer consoles)
- Legal / Compliance (ICO, DPA, tax)
- Job Hunt / Recruitment
- Personal / Property / Finance
- Other

Count approximate percentages per category. Identify the top 3 noise sources per inbox.

### 6. Cross-inbox comparison

Produce a comparison table with columns:
- Account
- Inbox volume (approx total)
- Noise level (Low / Medium / High / Very High)
- Top noise source
- Personal vs Business mix
- Actionability (low / medium / high)
- Maintenance priority

### 7. Write the report

Save a dated markdown file to:

```
~/.hermes/runbooks/inbox-analysis-YYYY-MM-DD.md
```

Structure:
1. TL;DR summary table
2. One section per account (size, unread/read sample, category breakdown with counts, observations)
3. Cross-inbox comparison table
4. Recommended actions (per account, concrete and ordered by priority)
   - Label creation with `kensei/` prefix
   - Auto-filter suggestions (by sender domain or `List-Unsubscribe` header)
   - Unsubscribe suggestions
   - Archive-or-delete targets
   - Infrastructure / project flags (e.g. paused Supabase projects)

### 8. Offer to execute filters

If the user approves, use the MCP tools to create labels and apply filters via `create_gmail_label`, `modify_gmail_message_labels`, etc.

### 9. Archive at scale (optional execution)

If the user issues a blanket directive like "archive all inbox items older than 30 days", use this high-throughput loop rather than one-by-one mutation:

1. **Search:** `search_gmail_messages(query="older_than:30d in:inbox", page_size=100)`
2. **Paginate:** Use `page_token` from the response. Repeat until exhausted or until the search returns fewer matches than the batch size.
3. **Batch-extract IDs:** Collect up to 50 `message_id` values at a time.
4. **Batch-archive:** Call `batch_modify_gmail_message_labels` with `remove_label_ids=["INBOX"]` (or whichever label is being archived).
5. **Repeat:** Continue through all pages. Report total archived after each batch.

**Tool guidance:**
- `batch_modify_gmail_message_labels` accepts up to 50 message IDs per call. This is the ONLY tool to use for bulk archive sweeps.
- `modify_gmail_message_labels` is for single-message mutations (testing, exceptions) and should NOT be used in a 100+ message archive.
- For 100 results per page, split into two 50-ID batches before calling.

## Pitfalls

1. **429 Rate Limits:** Fetching metadata for 100+ messages in parallel triggers Gmail API concurrency limits. Batch sequentially. Use `format=metadata` (no body) to reduce per-message payload.
2. **OAuth Port Contention:** In multi-Gmail probes, parallel calls across accounts can sometimes try to initiate OAuth and fail with `Port 8000 is already in use`. Retry the affected account sequentially before assuming token failure.
3. **Pagination Tokens Expire:** If context compresses between pages, you may lose the token. Fetch larger pages (100) and store the token if chaining multiple calls in one turn.
4. **Context Window Pressure:** 300-message listings with full metadata can easily blow the context window. Use `format=metadata` and aggregate category counts in your head rather than dumping raw Subjects into the response.
5. **Thread Duplicates:** `search_gmail_messages` returns message IDs. Some messages share thread IDs. Count unique threads if you care about conversation count vs raw message count.
6. **OTP / Magic Link Leakage:** Digest outputs must redact one-time codes, login links, and magic links. It is fine to say "OpenAI login-code email" or "Claude secure login link"; do not include code values or authentication URLs in the archive or outbound summary.
7. **Outlook Multi-Account Collection:** Outlook MCP calls are safest when every call includes the explicit `account` parameter and a restricted `$select` like `id,subject,from,toRecipients,receivedDateTime,bodyPreview,isRead,hasAttachments`. Verify returned `toRecipients` matches the queried account before trusting a cross-account digest.
8. **Context Pressure from `body_format=text`:** When fetching message content, `body_format=text` includes the full body text including Base64 encoded images, marketing HTML, and tracker pixels. A 25-ID batch can easily exceed 15,000 characters and may be truncated by context compression. **Only use `body_format=text` with 10 or fewer IDs per batch**. For metadata-only extraction (Subject, From, Date), you do not need `body_format=text` — the headers are already returned.

## Workspace Routing

When Sahil asks whether to generate Workspace prompts or execute mailbox setup directly:

- **Operations** is the default for mailbox agents. Mailbox digesting is a persistent responsibility, so model it as an Operations profile / standing agent.
- **Conductor** is for one-off setup, design, audit, or scaffold missions. Use it to research existing docs, generate the prompt pack, validate model policy, and create the implementation plan.
- **Swarm** is overkill for MVP mailbox digest unless there are multiple genuinely independent lanes.

Execution posture: do not ask Sahil to paste prompts back into Workspace when the agent can execute the safe path itself. Generate prompts internally, run safe checks, and report results. Stop for explicit approval before sending emails, deleting/archiving/labelling mail, or creating recurring token-spending jobs.

## Prompt Template (Operations)

When scaffolding a mailbox digest Operations agent, use this system prompt shape:

```text
You are KENSEI's mailbox operations agent for Sahil Saghir.

Mission: produce a safe, concise mailbox digest across connected Gmail and Outlook accounts.
Use metadata-first collection, redact OTPs/magic links/auth URLs, and never mutate mailboxes without explicit approval.

Accounts:
- Gmail: saghir.sahil@gmail.com, sahilsaghir.ss9@gmail.com, fusionfirststudios@gmail.com
- Outlook: sahil_ss9@hotmail.com, sahil_saghir@hotmail.co.uk, sahil_ss@outlook.com, matchdaymaestro@outlook.com

Model policy: default worker model ollama-cloud / kimi-k2.6.

Routine:
1. Probe account availability with explicit account parameters.
2. Search recent inbox items using metadata-only or restricted select fields.
3. Categorise into urgent, job hunt, family/personal, project/admin, finance/property, subscriptions/noise, and security/auth.
4. Redact OTPs, magic links, and auth URLs.
5. Return a short digest with actions needing Sahil and suggested safe follow-ups.
6. Stop before any mailbox mutation.
```

## Related Skills

- `add-gmail-account` -- If a Gmail account is missing or token is expired.
- `google-workspace-mcp` -- For OAuth troubleshooting, rate-limit-safe batch operations, and the `category:*` search strategy reference.
- `google-workspace` (productivity) -- If you need to use the `gws` CLI instead of MCP tools.
- `mailbox-digest` (autonomous-ai-agents) -- The consolidated mailbox agent skill with profile creation and scheduling patterns.
