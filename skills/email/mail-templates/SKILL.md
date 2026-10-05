---
name: mail-templates
description: Draft Mail Studio templates from real scrape fields.
version: 1.0.0
author: IntelliVerse X
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Email, Mail Studio, Templates, Outreach]
    related_skills: [engagement-stack]
---

# Mail Templates

Draft and save Mail Studio templates for the signed-in brand. This skill writes the template. It does not send mail. Sending happens only when the user approves a run in Automation Studio.

Talk to the user with the product name Mail Studio. Do not ask them to paste a mail key if this brand already saved mail settings.

## When to Use

- "Create a mail template."
- "Write a follow-up email for this business."
- "Change the cold email this brand sends."
- "What fields can this template use?"

Do not use this to send a blast, scrape leads, or send SMS. Those belong to `engagement-stack`.

## Prerequisites

The user is signed in to a brand. Mail Studio is available for that brand. Template tools ride the signed-in session. If mail is not set up, say so and stop. Do not invent a from-address or a workspace.

## How to Run

1. Ask what this mail is for, how it should look, and which scrape facts belong in it.
2. Use only facts listed in `engagement-stack` under "What the scrape returns."
3. Draft the subject and body. Leave a missing fact blank.
4. Show the draft and wait for a yes before saving it on the business.
5. Tell the user that preview does not send, and mail leaves only on Approve.

## Procedure

### 1. Ask before writing

Ask, in one pass:

- Which business, and which step: first touch, follow-up 1, follow-up 2, or breakup.
- The context: why we are writing, and what we want them to do.
- How it should look: length, tone, and the sections (greeting, one proof line, ask).
- Which details fit. Offer only the scrape facts from `engagement-stack`. The user picks. Do not add a fact they did not pick.

Done when those four answers are written down.

### 2. Fill only real fields

Inner lines use the picked facts, and only when that scrape returned a value. Name, business, category, city, website, rating, review count, last post, and booking link are the usual ones. If a picked fact is empty for a lead, leave that line blank. Never write "Not found in this scrape." Never invent a last-post date, a competitor, or a review count.

Done when every sentence is either static copy the user approved or a picked field.

### 3. Save after a yes

Show subject, body, and which fields are filled. Save only after the user agrees. List existing template ids with `automation_studio_list_templates`. Store the draft in Mail Studio, then attach that id on the business with `automation_studio_save_business` (`templateId` for the first mail, `fu1TemplateId`, `fu2TemplateId`, or `breakupTemplateId` for later steps). To change the From address, call `notifuse_workspaces_update_email_senders` with the current integration name and `provider.senders` only (`email`, `name`, `is_default`). Do not send SES or SMTP keys. Done when the template is stored on that business, or the user declined.

### 4. Do not send

Creating or editing a template does not send mail. Preview in Automation Studio also does not send. Mail leaves only when the user approves that run.

## Pitfalls

- A listed template is not a sendable mail until it is the template on that business.
- Missing scrape facts stay blank. Do not substitute a guess.
- Do not ask the user to paste a mail key that this brand already saved.

## Verification

Read the saved template back. The subject and body match the approved draft, blank fields stay blank, and no mail was sent.
