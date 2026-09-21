# Gmail Inbox Audit Reference (Category Search Strategy)

## When to Use This Reference

- During a multi-account inbox triage where you need a fast category breakdown (Promotions, Social, Updates, Purchases, Forums) without parsing 100+ individual message subjects.
- When `get_gmail_messages_content_batch(format="metadata")` would otherwise produce an unreadable wall of text for large inboxes.

## Category: Search Queries

The Gmail `search_gmail_messages` API accepts category filters as query strings. The exact syntax is:

```
in:inbox category:promotions
in:inbox category:social
in:inbox category:updates
in:inbox category:forums
in:inbox category:purchases
```

Other valid categories that may be present but were not tested:
- `category:primary`

## Execution Order for Multi-Account Audit

1. For each account, run the 5 category searches above with `page_size=1`.
   The tool output starts with `Found N messages matching '...'` — extract that count.
2. If any category returns >0, note: Gmail tabs ARE working for this account.
3. If ALL 5 return "No messages found", that does NOT mean there are zero emails in that category. It means **the account has 0 emails matching that query**, which, when the inbox clearly has messages, means **category classification is not routing emails into those buckets**.
   - Typical cause: Gmail tabs are present (`CATEGORY_PROMOTIONS` exists as a label) but the Inbox type is **NOT** set to "Default" (Tabs) in Settings → Inbox.
   - All emails land in Primary instead.

## Interpreting Results

| Category | Typical Noise | Actionable? | Typical Label |
|---|---|---|---|
| Promotions | Marketing drips, affiliate emails, discount codes | Low | `kensei/Newsletter` |
| Social | Skool digests, community updates, social media | Low | `kensei/Skool` or `kensei/Social` |
| Updates | Platform notifications, service alerts | Medium | `kensei/Updates` |
| Purchases | Receipts, order confirmations, shipping | Yes | `kensei/Financial` |
| Forums | Mailing lists, Google Groups | Low | `kensei/Forums` |

## Batching Rules for Content Extraction

If a representative sample of Subjects/Senders/Date is still needed, do NOT dump 100 message IDs into `get_gmail_messages_content_batch`.

1. Fetch a list of IDs via `search_gmail_messages(query="in:inbox", page_size=25)` per account.
2. Split into **10 IDs per `get_gmail_messages_content_batch` call**.
   - 25 IDs at once triggers HttpError 429 / context bloat.
   - 10 IDs keeps response size under ~5KB with `body_format=text`.
3. Parse only Subject, From, and Date fields from the output.
   Use string splits on `Subject:`, `From:`, `Date:` rather than trying to JSON-parse the MCP response text.
4. If a duplicate set of IDs is requested, the server seems to deduplicate internally. Track IDs yourself if you need strict uniqueness counts.

## Pitfalls

- **Gmail API category indexing lag:** Emails sent minutes ago may not show up in `category:*` searches yet. Use for audits of backlog, not real-time eventing.
- **False negatives:** `category:forums` is often empty even on accounts with lots of mailing-list traffic because those emails land in Updates instead.
- **Context pressure:** `body_format=text` extracts full message bodies including Base64-encoded images, tracker pixels, and long marketing footer HTML.
  - Always use `body_format=text` with **small batches**.
  - Never use it for a 100-ID batch in a single call: the response can exceed 15,000 characters and may be truncated or break context compression.

## Related

- `gmail-inbox-audit` skill for the full report-generation workflow.
- `google-workspace-mcp` `SKILL.md` for rate-limit safe batch operations and label-naming conventions (`kensei/` prefix).
