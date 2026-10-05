---
name: engagement-stack
description: Run scrape, preview, and approve in Automation Studio.
version: 1.0.0
author: IntelliVerse X
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Automation Studio, Mail Studio, CRM, SMS]
    related_skills: [mail-templates]
---

# Engagement Stack

Run the signed-in brand's outreach from chat: scrape, preview, approve, follow up, text, and call. Automation Studio is what the user sees. n8n is the engine under it. Do not ask the user to open n8n or paste a key this brand already saved.

Mail copy belongs to `mail-templates`. This skill decides when a run may send.

## When to Use

- "Scrape businesses for this brand."
- "Show me the preview before anything sends."
- "Approve this run" or "do not send."
- "Text these leads" or "call this number."
- "Why did the scrape stop?"

## Prerequisites

The user is signed in to a brand. Automation Studio tools are connected for that brand: `automation_studio_status`, `automation_studio_preview`, `automation_studio_approve`, `automation_studio_cancel`, `automation_studio_pause`, `automation_studio_resume`, `automation_studio_send_followup`, `automation_studio_send_sms`, `automation_studio_place_call`, `automation_studio_save_business`, `automation_studio_remove_business`, `automation_studio_list_runs`, and `automation_studio_list_templates`.

If a tool is not connected, say which one and stop. Do not fall back to a shared key.

## How to Run

1. Call `automation_studio_status` for this brand.
2. Scrape or preview with `automation_studio_preview`. Show the leads. Send nothing.
3. Send mail only after the user says yes, with `automation_studio_approve`.
4. Texts use `automation_studio_send_sms`. Calls use `automation_studio_place_call`. Numbers come from the CRM person. The text comes from the business message, with name and business filled in.
5. Follow-ups use `automation_studio_send_followup` and the template for that step.
6. Add or update a business with `automation_studio_save_business`. Remove one with `automation_studio_remove_business`. Past scrapes and their leads are `automation_studio_list_runs`. Template ids are `automation_studio_list_templates`.

## Procedure

### 1. Check status first

`automation_studio_status` tells you if the brand can scrape, mail, text, or call. If mail credit is empty, say the wallet is empty. That is not a broken studio. Done when the user knows what is ready.

### 2. Preview, then wait

`automation_studio_preview` saves leads. It does not send mail. If no leads qualify, say there are no leads and offer another scrape or cancel with `automation_studio_cancel`. Do not show Approve when there is nothing to send.

People land in CRM with phone, company, and social on their own fields.

### What the scrape returns

Apify reads the public business profile. These are the only facts a template may use. `mail-templates` asks the user which of these belong in the mail.

| Fact | Use in a template |
|---|---|
| Owner or contact name | Greeting, when the scrape has a name |
| Business name | Subject and body |
| Category | What kind of business |
| City, address | Location line |
| Website | Only if a site was listed |
| Phone | SMS and calls, from the CRM person |
| Email | Who the mail is to |
| Google rating, review count, photo count | Only when the number is present |
| Last post date, days since last post | Only when a post date was found |
| Booking link | Only when a booking page was found |
| Facebook, Instagram | Only when that profile link was found |

If a cell is empty, leave it blank. Do not invent a competitor, a last-post age, a review count, or a booking page. Do not print "Not found in this scrape."

### 3. Stop cleanly

If Apify has no credits, say that and stop the scrape. If the engine stops before it calls back, say the run did not finish and do not claim the mails were sent. Do not keep retrying a scrape that already reported a stop reason.

### 4. Send only on approve

`automation_studio_approve` is the send. Say who will get the mail before you call it. Pause with `automation_studio_pause` and resume with `automation_studio_resume` when the user wants the schedule held.

### 5. Text and call

Take the phone from the CRM person. Take the SMS body from the message saved on that business. Call `automation_studio_send_sms` or `automation_studio_place_call`. Do not ask for the Telnyx key again when this brand already saved it.

## Pitfalls

- Preview is not Approve. Mail leaves only on Approve.
- n8n is the engine. The user works in Automation Studio.
- A daily cap is how many new businesses this run may take. It does not shorten the gap between follow-ups.
- Do not invent a phone, a from-address, or a template the business does not have.

## Verification

After a preview, no mail has been sent and the lead count matches the tool result. After an approve, the tool result names the send. After a stop, the user heard the reason and the run is not still going.
