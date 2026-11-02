# WhatsApp Multi-Point Response — Landowner/Partner Queries

## When to use this reference

The user asks you to draft a WhatsApp response addressing **multiple points raised by a business partner** (landowner, co-developer, investor, CA) — typically after receiving a message with several objections, questions, or proposed changes to a sharing agreement / JDA / SSA. The partner's message covers 5-15 distinct topics; each needs a factual response in a single coherent message.

**Trigger phrases:** "draft a WhatsApp response", "point-by-point feedback", "my response to his points", "address each of these".

## Workflow

### Step 0 — Receive the partner's original message (may arrive AFTER or BEFORE the voice note)

The partner's written message can arrive at two different times, and the workflow differs:

**Scenario A — Partner message first, then voice note (common):**
1. Read partner's message → identify each distinct point raised
2. Listen to user's voice note → map each feedback point to the corresponding partner point
3. Draft responses using the user's corrections for each point

**Scenario B — Voice note first, then partner message (this session):**
1. Parse user's voice note into distinct feedback topics (the user often says "I'll share the original message below")
2. When the partner's message arrives, **remap** each voice-note feedback to the partner's corresponding numbered point
3. The partner's points may be numbered differently than the user's speaking order — re-index to match the partner's numbering
4. The partner may reference specific figures (e.g. "~641 sft discrepancy", "Section 2(k) of RERA Act") — mirror these specifics in the response to show you read their message carefully
5. The partner's message structure should drive the response structure (match their point numbers), not the user's speaking order

**Why this matters:** The user's voice note jumps between topics in their own mental order. The partner's message has a deliberate numbered structure. If you follow the voice-note order instead of the partner's numbering, the recipient sees a confusing reordering of their questions. Always re-index.

### Step 1 — Parse the user's voice note / instructions

The user typically provides their feedback in a **single long voice note** covering multiple topics in sequence. Do NOT ask for clarification on each point — extract all distinct points from the voice note:

- Listen for topic shifts ("Coming to the next point...", "Next was...", "On the issue of...")
- Each point usually has: (a) what the partner raised, (b) the user's correction/clarification, (c) the position to communicate
- Note any points where the user says "I'll address this separately" — those get their own message

### Step 2 — Structure the response

Each point has two parts: a **brief keyword summary of what the other person raised**, followed by **the user's response**.

```
━━━━━━━━━━━━━━━━━━━━━━━━
[Number]/ *[3-5 keyword phrase summarizing their point]*

[User's 2-4 sentence response]
━━━━━━━━━━━━━━━━━━━━━━━━
```

**Rules for the summary line:**
- Use 3-5 keywords that describe the other person's point, not the response (e.g. "Gratitude for school fees & basic support" not "Nishant's reassurance")
- Number the response points — they may or may not align with the original message's numbering
- The summary helps the reader orient before reading the response
- Bold the entire summary phrase

**Example (from Prakash employee message session, Jun 2026):**

Employee's points (numbered 1-7 in his message):
- Point 1: "grateful for school fees / house / support"
- Point 2: "priority is to deliver assignments"
- Point 3: "BD on hold → financial dilemma → family emergency"
- etc.

Response format:
```markdown
━━━━━━━━━━━━━━━━━━━━━━━━
1/ *Gratitude for school fees & basic support*

[Response text...]
━━━━━━━━━━━━━━━━━━━━━━━━
```

Format each point with:

```
━━ *[Point Number]. [Topic Heading]* ━━
[2-4 sentence response — factual, polite tone]
• Bullets for lists (max 3-4 per point)
• Bold for emphasis on key terms
```

**Rules:**
- Use `━━` separator lines between points — they render cleanly on WhatsApp
- Bold the point heading with `*asterisks*`
- Keep each response to 2-4 sentences maximum
- Use bullet points (•) for structured items
- **No indentation** — WhatsApp doesn't support blockquotes or nested indentation reliably
- **No sender signature** — the phone number identifies the sender
- End with a single closing line

## Tone calibration

| Partner type | Tone | Example opener |
|---|---|---|
| Landowner (JDA partner) | Respectful, factual, deferential on disputed points | "Thank you for your message. Point-wise below:" |
| Co-developer / Investor | Collaborative but firm on scope | "Appreciate the detailed feedback. Here's our position:" |
| CA / Legal advisor | Precise, clause-referencing | "References in line with the JDA below:" |

- **Polite but NOT deferential when the facts are on your side** — use "Just to correct —" or "You may recall we discussed this" rather than "I'm sorry"
- **For repeated clarifications** (user says "I've explained this 10 times"), lead with: "Repeating myself but — [position]"
- **No aggressive language** — avoid "you're wrong", "you misunderstand". Use "happy to clarify", "just to correct"

### User-specific: Nishant (NDR) — "constructive, positive, cooperative" tone

When drafting WhatsApp responses for Nishant to landowners/business partners (notably Raghu Iyer for Ranka Amber):

- **Default tone is warm-collaborative, not cold-factual.** Open with appreciation ("Thank you for your thorough review — really appreciate the time spent on this"). Close with forward-looking goodwill ("Looking forward to closing this out together").
- **Replace defensive phrasing** with collaborative alternatives:
  - "I must clarify" → "I think there may be some miscommunication here. To be transparent —"
  - "There is no" → "Since there's no commitment, there's nothing to record"
  - "This is important to clarify" → "Just to share how we usually approach this"
  - "There is no legal requirement" → avoid; explain WHY it works instead
- **Use "we" language** framing the response as a joint effort: "We'd love to include all those spaces, but unfortunately..." / "Consider it a good-faith effort from our side to maximise the project's value for both of us."
- **Concede gracefully** where the partner is right: "You're right — the '+4' reference was an error which I'll correct."
- **Offer goodwill documentation** when the partner wants something recorded that isn't required: "That said, if it gives you comfort, I'm happy to add a simple recital stating..."

This applies to ALL WhatsApp responses to Raghu Iyer / Ranka Amber and other JDA landowners by extension. Use this tone table instead of the generic one when drafting for Nishant.

## Delivery — Code block vs HTML file

### Single message: code block

Wrap the entire message in a ``` code block in your Telegram response. The user copies and pastes into WhatsApp. WhatsApp converts markdown (`*bold*`, bullets) natively.

### Multiple messages (2+): HTML file with copy buttons

When the user asks for **more than one separate WhatsApp message** (e.g. a main response + a follow-up on a specific topic), **do NOT send them as separate code blocks in Telegram**. Telegram splits them across messages and the user can't easily copy both.

Instead:
1. **Create an HTML file** with both messages as separate code blocks, each with a JavaScript `Copy to clipboard` button
2. Include a clean card layout — one card per message with header, body, and copy button
3. **Send the file via Telegram** using `send_message` with `MEDIA:<path>`:
   ```
   write_file(path="/opt/data/whatsapp-{project}-{topic}.html", content=html_string)
   send_message(message="Here's the file — MEDIA:/opt/data/whatsapp-{project}-{topic}.html")
   ```
4. The user opens the HTML in their browser and clicks Copy for each message, then pastes into WhatsApp

**HTML file structure (minimal):**
- One `<div class="card">` per message
- Each card has a `<pre class="code-block">` containing the message text (preserve WhatsApp markdown formatting)
- Each card has a `<button onclick="copyText(...)">Copy Message</button>`
- JavaScript `copyText` function uses `navigator.clipboard.writeText()` with a `document.execCommand('copy')` fallback
- A toast/banner shows "Copied!" on success
- Green tip card noting WhatsApp preserves bold/formatting on paste

**Trigger proactively:** If the user's instructions produce 2+ separate message threads, generate the HTML file immediately without asking whether they want code blocks or HTML.

### Step 5 — Separate messages for separate topics

If the user says a particular topic needs a *separate* message (e.g. Premium FAR clause reference from a JDA), break it out as a second code block with its own heading.

## JDA / Clause Research for Disputed Points

When the partner's point references a specific JDA clause (e.g. Premium FAR cost, sharing ratio, marketing fee), and the user says "look up the JDA" or "check the clause":

### Step 1 — Locate the JDA documents in Drive

From prior session references, find the Drive file IDs:
- **JDA Registration** (Principal Agreement) — scanned PDF, typically 50+ pages
- **Addendum(s)** — may be Google Docs or PDFs

Search session history (`session_search`) and existing references for IDs. Common reference files store these under document ID tables (e.g. `ranka-amber-audit-may2026.md`).

### Step 2 — Download and OCR

Use `delegate_task` with `tools.gws_auth.build_service("drive", "v3")` to download the PDFs (or `tools.gws_dwd_tools` if a file is reachable only under ndr@draas.com's identity — `tools.gws_sa` does not exist, removed 2026-05-08), then:
- For **scanned PDFs**: OCR with Tesseract (`pytesseract`) after converting PDF pages to images (`pdf2image` or `PyMuPDF`)
- For **native PDFs** (text-based): extract directly with `PyMuPDF` (`fitz`)
- For **Google Docs**: export as plain text by setting `mimeType='text/plain'` in `drive.files().export()`

The subagent should be given the exact Drive IDs and told to find all references to the specific clause topic (e.g. "Premium FAR", "Premium FSI", "TDR", "additional SBUA", "marketing fee").

### Step 3 — Extract relevant clause text

From the OCR output, locate the specific clause (e.g. Clause 5.6) and extract its exact wording. Key details to capture:
- Who bears cost?
- How is additional area shared?
- What rights are covered?
- Is utilisation mandatory?
- Does the Addendum modify this clause? (Check "Remaining Terms Unchanged" clause in Addendum)

### Step 4 — Apply to the current situation

Present the clause findings in the WhatsApp response structured as:
- What the clause says (in plain language, bullet-pointed)
- Why it does (or doesn't) apply to the current situation
- The user's position (e.g. "Clause 5.6 is not triggered because no Premium FAR was purchased")

### Pitfall — `tools.gws_sa` does not exist, and there is no local token file either

Two dead references retired 2026-05-08, both still worth knowing about so you don't reach for them:
- `tools.gws_sa` — module removed entirely, raises `ModuleNotFoundError`. `GOOGLE_SA_KEY` is n8n-only, never present here.
- `/data/hermes/google_token.json` — does not exist on this deployment (confirmed 2026-09-19). Do not read a raw token file off disk under any circumstance; credentials live only in the `gws-vault` daemon.

**Fix — use the sanctioned paths:**
- Acting as the session's own account: `tools.gws_auth.build_service('drive', 'v3')` — works the same inside `terminal()` subprocesses and `delegate_task()` subagents (it routes through the RPC-mediated vault fetch automatically when sandboxed).
- Acting as ndr@draas.com specifically (impersonation): `tools.gws_dwd_tools` — admin-gated DWD path, returns data only, never a credential.

**For JDA research specifically:** Use `delegate_task` with instruction to call `tools.gws_auth.build_service('drive', 'v3')` (or `tools.gws_dwd_tools` if the file requires ndr@draas.com's identity specifically) — do not reference a local token file path.

### Pitfall — SSA cannot exceed what is legally sanctioned

When the partner asks you to record amenities or spaces in the Sharing Agreement that go beyond the building sanction (e.g. gym equipment in stilt area, rooftop amenities), the response pattern is:

> "The [area] has only [X] as per the sanctioned plan — and our Sharing Agreement cannot go beyond what is legally sanctioned. However, we are happy to record that:
> • The common areas will be finished to premium specifications
> • We will be placing [Y] in [area] for residents' use
> This is the best we can do within what is legally permissible."

This applies to:
- **Stilt area amenities** (common toilet → premium spec + gym equipment as goodwill, not commitment)
- **Rooftop amenities** (no built-up area possible under sanction → only potted plants, seating, clothes drying, solar water heaters, small steel extension)
- **Common area accounting** (only FAR-eligible spaces qualify → non-FAR areas explained to customers separately)

### Point type — "Legally impermissible in SSA"

When the partner asks for something that the SSA cannot legally record:

1. State the legal limitation clearly but politely: "Our Sharing Agreement cannot go beyond what is legally sanctioned."
2. Offer what CAN be recorded: amenities, premium specs, design intent
3. If the item is still a grey area (not blatantly illegal), offer to mention it as an informal note
4. Do NOT fabricate commitments that don't exist in the sanction plan

### User-specific: Repeated clarifications

When Nishant has already explained something multiple times (e.g. rooftop amenities — "clarified this at least three times in earlier messages"), use this softer pattern instead of an exasperated tone:

> "I think there's been some miscommunication on this, and I've clarified this at least three times in our earlier messages in this group, so let me restate clearly:"

Then follow with what IS being provided, not just what ISN'T. Structure as two clear sections (e.g. *Rooftop Terrace:* and *Stilt Area:*) with bullet points for specifics.

### User-specific: Partnership/goodwill framing for cost disputes

When Nishant needs to explain that costs were absorbed rather than passed on (e.g. Premium FAR), frame it as partnership goodwill:

> "When we started this project together, we could have built to [baseline] and walked away. That would have been lower cost for us. But once we are in it together, we are in it together. Through provisions in law and relevant circulars — fully legal — we achieved [result]. Our costs went up, but we didn't come back to you because on good faith we chose to absorb that."

Then distinguish the current situation from what the JDA clause says — explaining why the clause is "not triggered" rather than arguing about who should pay.

### HTML file delivery — Fallback when MEDIA blocked

If `media_delivery_allow_dirs: []` in the gateway config prevents MEDIA delivery of HTML files, present each message as a standalone code block in the Telegram response with clear copy instructions. The user copies each block and pastes directly into WhatsApp — the formatting (bold, bullets, emoji) is preserved on paste.

### Pitfall — Drive file IDs can change

The Addendum file ID from earlier references may return 404. If so, search Drive by filename using `drive.files().list(q="name contains 'Addendum' and name contains 'Raghu Iyer'")` to find the current file ID.

## Common point types and their response structure

### RERA / Regulatory clarifications
*If partner says "RERA says X"...*
> Lead with: "RERA has nothing to do with the Sharing Agreement. RERA is marketing permission. The Sharing Agreement governs what is sanctioned."

### Discrepancy / Correction
*If partner flags a discrepancy that is actually standard practice...*
> Lead with the corrected fact, then the explanation: "The sanction software uses half-wall for internal walls only. External walls take full area. This is standard practice."

### Commitment denial
*If partner asks for a commitment that doesn't exist...*
> "I'm repeating myself but — [clear denial]. [What we ARE providing instead]."

### JDA clause reference
*If partner misreads a JDA clause...*
> Quote the clause number, what it actually says in plain language, then apply to the current situation.

## Example format (from Ranka Amber session, June 2026)

```
━━ *1. Architect's RERA Compliant Area Statement* ━━
The architect's area statement has already been shared. Happy for the architect to put it on his letterhead, sign and stamp. It covers:
• RERA Carpet Area
• Exclusive Attached Balcony
• Super Built-Up Area (SBUA)

━━ *2. RERA Carpet vs Marketing Carpet Discrepancy* ━━
You may recall we discussed this. RERA Carpet excludes balconies but includes internal walls. Marketing carpet was the opposite. We will only use three figures in all marketing — RERA Carpet, Attached Balcony, and SBUA.
```

## Pitfalls

1. **Don't repeat the partner's message verbatim** — the partner already knows what they said. Just reference the topic.
2. **Don't be verbose** — each point should be 2-4 sentences. Longer means the point has sub-points.
3. **Don't include a sender sign-off** — the phone number identifies the sender.
4. **Don't use `>` blockquotes** — WhatsApp doesn't support them. Use bullets (•) or separator lines.
