# Property Inventory — External Sales Head Communication

> **Scope note:** this reference covers **one project's** unit inventory to a single sales
> head. For a **multi-asset portfolio assembled for liquidation / monetisation** (inventory
> mined out of scattered historical emails, every collateral linked, rates benchmarked
> against live portals, delivered as a self-contained HTML email body) use the
> **`asset-liquidation-dossier`** skill instead — it carries NDR's two standing corrections
> for that class: never flat prose, and the HTML **is** the email body rather than a PDF
> attachment. See also `gmail-draft-formatting-pitfalls.md` § "Scope correction" — rich HTML
> in `add_alternative` multipart drafts renders fine; the minimal-HTML rule is for short
> threaded replies only.

## When to use

The user asks you to send structured **unit inventory data** (a list of units they own in a project) to an **external sales head / channel partner** who will sell the units. Delivery is via **both** WhatsApp (immediate) and email (formal with HTML table).

**Not for:** Internal sales team coordination (use `sales-inventory-coordination-whatsapp.md`) or lead follow-ups (use `whatsapp-sales-leads-followup.md`).

## Canonical example (Century Regalia → Antony, Century Real Estate, Jul 2026)

**Context:** Nishant owns 5 units in Century Regalia (Kodihalli, Bangalore) — some jointly with Kanta Ranka/Mamta Rathod, some through South City Properties Pvt Ltd. Wanted to share inventory with Antony (Century Real Estate Sales Head) and request a minimum rate of Rs 27,500/sft take-home from US brokerage.

## Workflow

### 0. Find the source data

The user says "there's an email with all that content" or "check the email I sent about [project]":

1. Search Gmail sent items for the reference email by project name, date range, or recipient
2. Extract the HTML body (handle multipart/mixed recursively, base64 decode)
3. Parse table data from `<table>` elements in the HTML — extract rows, cells, headers
4. Present the extracted data to the user for confirmation before proceeding

See `email-drafter` → `references/existing-email-data-extraction.md` for the full Gmail extraction and HTML table parsing pattern.

### 1. WhatsApp — immediate delivery

#### WhatsApp formatting for property unit inventory

Structure the message with WhatsApp markdown:

```
*Project Name — Our Unit Inventory*

Hi [Name], here are the units we hold in [Project] ([Location]).

*Owned by [Owner Group 1]:*
*Unit Name/Wing + Number* — [Type], [Floor] | [Names of owners] (Joint/Sole)
  Carpet: [N],NNN sft | Super Built-up: ~N,NNN sft

*Owned by [Owner Group 2]:*
*Unit Name/Wing + Number* — [Type], [Floor]
  Carpet: [N],NNN sft | Super Built-up: ~N,NNN sft

Total: N units

Note: [Any key info — e.g. floor can be combined, unique features]

Request: After discussion, we request you to achieve a minimum take-home rate of Rs [N,NNN]/sft from [source] brokerage. Brokerage can be on top.

If you feel this rate is hard to push, happy to have a discussion.

Regards,
[Nishant]
```

**Formatting rules:**
- Project heading in bold as the first line `*Project Name — Unit Inventory*`
- Ownership groups as bold sub-headings `*Owned by [Group]:*`
- Each unit's identifier (Wing + Number) in bold `*Crissa 401*`
- Sub-details indented below each unit: Carpet area and Super Built-up area
- Numeric values with thousand separators: `1,719 sft`
- Total count line before the request
- Note line if relevant (e.g. units on same floor can be combined)
- Request/ask clearly separated at the bottom
- Use `api.whatsapp.com` URL format (preferred over wa.me)

**Length check:** A 5-unit message with all details typically fits in a wa.me deep link. If it exceeds ~4,000 chars encoded, switch to HTML card delivery (see `whatsapp-drafter-full.md` → HTML card section).

#### Source data from Gmail

Not a shared sheet — the inventory data came from a previously-sent email. The table data in that email was the source. See `email-drafter` → `references/existing-email-data-extraction.md`.

### 2. Email — formal with HTML table

After the WhatsApp message, send the formal email with an HTML table.

#### HTML table format for property inventory

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 11pt; }
    th { background-color: #1a3a5c; color: white; padding: 8px; text-align: left; }
    td { padding: 6px 8px; border-bottom: 1px solid #ddd; }
    .section-header { background-color: #e8edf2; font-weight: bold; padding: 6px 8px; }
    tr:hover { background-color: #f5f5f5; }
  </style>
</head>
<body>
  <p>Hi [Name],</p>
  <p>As discussed, please find below the details of our unit inventory in [Project] ([Location]) for your reference.</p>

  <p><strong>Project:</strong> [Project Name]<br>
  <strong>Promoter:</strong> [Promoter Entity]<br>
  <strong>Location:</strong> [Area, City]</p>

  <table>
    <tr>
      <th>#</th>
      <th>Wing</th>
      <th>Unit</th>
      <th>Allottee(s)</th>
      <th>Type</th>
      <th>Floor</th>
      <th>Carpet (sft)</th>
      <th>Super Built-up (sft)</th>
    </tr>
    <tr>
      <td colspan="8" class="section-header">Owned by Nishant Ranka &amp; Family</td>
    </tr>
    <tr>
      <td>1</td>
      <td>[Wing]</td>
      <td>[Unit]</td>
      <td>[Owners — specify Joint/Sole]</td>
      <td>[N Bed]</td>
      <td>[Floor]</td>
      <td>N,NNN</td>
      <td>~N,NNN</td>
    </tr>
    <!-- more rows -->
    <tr>
      <td colspan="8" class="section-header">Owned by [Entity Name]</td>
    </tr>
    <!-- more rows -->
  </table>

  <p>After internal discussion, we request you to kindly achieve a minimum take-home rate of <strong>Rs [N,NNN]/sft</strong> from [source] brokerage. Brokerage can be on top.</p>

  <p>If you feel this rate is hard to push, we are happy to have a discussion.</p>

  <p>Regards,<br>Nishant Ranka</p>
</body>
</html>
```

**Key details to include in the email body (outside the table):**
- Project name, promoter entity, location
- Ownership overview (who owns which units)
- Total number of units

**Rate request wording:** Follow the user's exact phrasing. Canonical pattern from Jul 2026:
> "After internal discussion, we request you to please achieve a rate of Rs 27,500 minimum take-home from US brokerage. Brokerage can be on top of it. If you feel this rate is hard to push, happy to have a discussion."

### 3. CC pattern

When sending to an external sales head, CC the sender (self) and the co-owner/partner:
- **Self:** ndr@draas.com
- **Co-owner/partner:** mdr@drahomes.in (Manish Rankar) — add only when the user explicitly names them

### 4. Contact management — add before sending

When the user provides a new contact's email, add them to **both** stores before sending:
1. **Google Contacts** (People API) — create a contact with name, email, phone
2. **NDR DRAAS Contacts Sheet** — append a new row at the bottom with name, email, phone (see `contact-phone-lookup` → `references/contact-creation-dual-flow.md`)

### 5. Send (not draft)

Unlike the standard drafts-only rule for email, this pattern is for **external coordination emails the user explicitly asks to send**. The user says "send it to Antony" — not "draft it". The explicit "send" override applies because this is a time-sensitive business proposal.

**Exception rule:** Only send without draft-review when:
- The user gives you the exact wording / content to include
- The user says "send this email" (explicit send instruction)
- The email content is a straightforward presentation of factual data (not a draft needing negotiation review)

### Pitfalls

- **Don't skip WhatsApp** — The user wants BOTH WhatsApp AND email. WhatsApp is for immediacy; email is for reference.
- **Don't ask which format** — Default: WhatsApp with markdown bold/bullets + email with HTML table. The user expects both.
- **Contact may not exist in any store** — The external sales head may only have a phone number stored, not an email. Explain what you found and ask for the missing field rather than searching endlessly.
- **Brokerage wording** — The user may frame it as "X minimum take-home from US brokerage" meaning the developer takes home X net, and brokerage (sales commission) is paid on top. Follow their exact phrasing.
- **HTML email encoding** — Use proper HTML entities: `&amp;` for `&`, `&lt;` for `<` in the HTML table content. Test that `&amp;` renders correctly in Gmail.
