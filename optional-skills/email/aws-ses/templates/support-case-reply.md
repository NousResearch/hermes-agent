---
template: aws-ses-production-access
description: "Complete AWS Support case templates for SES production access: initial request + follow-up reply."
version: 2.0.0
---

# AWS SES Production Access — Complete Support Case Templates

**Use these templates when:**
- **Part A (initial request):** you are opening a NEW production-access case for the first time.
- **Part B (follow-up reply):** your request was **DENIED** (or stuck PENDING) and you must reply to the **existing** case. Do NOT open a new case — the API/console will return `ConflictException` and the new case will be closed as a duplicate.

Fill in every `{{placeholder}}` with your real information. Be specific — vague answers are the second most common rejection reason after wrong mail-type classification (see `SKILL.md` Step 4).

---

# Part A — Initial Production Access Request

**Use when:** opening the case via the AWS Support Center (create case → Service: SES → Category: Production Access).

**Subject:** Production Access Request — {{website_url}} — {{mail_type}} emails

**Body:**

Hello AWS Support,

We are requesting production access for Amazon SES to send
**{{mail_type}}** email from the verified domain **{{verified_domain}}**.

## Mail Type

We selected **{{mail_type}}** because our emails are
{{justification_for_mail_type}}.

*(Example justifications)*
- For MARKETING: "…a weekly newsletter sent to subscribers who have explicitly
  opted in to receive updates about our product."
- For TRANSACTIONAL: "…system-generated password-reset emails and order
  confirmations triggered by user actions on our platform."

> ⚠️ **Critical:** classify honestly. Marketing/newsletter/bulk sends MUST be
> MARKETING. The #1 rejection cause is requesting TRANSACTIONAL for what is
> actually marketing/bulk email.

## Use Case & Website

Our website **{{website_url}}** is
{{description_of_what_the_site_does_and_how_users_sign_up}}.

We send the following types of email:
1. {{email_type_1_description}}
2. {{email_type_2_description}}

## Expected Volume

We expect to send approximately **{{expected_volume}}** emails per
{{day|week|month}}, growing gradually as our user base expands. Our initial
volume is well within the default production limit of 50,000 emails per 24 hours.

## Email Address Collection & Consent (Opt-In)

Every recipient on our list has provided **explicit consent** through the
following mechanism:

- **Opt-in method:** {{opt_in_mechanism}}
  *(Example: "Users subscribe via a sign-up form on our website. They receive
  a confirmation email (double opt-in) and must click the verification link
  before any further email is sent. We do not purchase, scrape, or rent email
  lists.")*
- **Consent records:** We maintain timestamps and IP addresses of opt-in
  confirmations in our database for audit purposes.

## Bounce & Complaint Handling

We process hard bounces automatically:
- SES sends bounce notifications to an **SNS topic** configured on our verified
  domain identity.
- Our application subscribes to this SNS topic and **automatically suppresses**
  any address that generates a hard bounce.
- SES sends complaint notifications to the **same SNS topic**; our application
  **immediately and permanently suppresses** any address that generates a
  complaint.

## Unsubscribe / Opt-Out

Every marketing email we send includes:
- A **one-click unsubscribe link** in the footer that instantly removes the
  recipient from all lists.
- A **physical mailing address** (as required by CAN-SPAM).
- Clear sender identification (our brand name and verified domain).

Unsubscribe requests are processed **immediately** and take effect for all
future sends.

## Compliance

We comply with:
- **CAN-SPAM Act** — physical address, clear sender identity, working
  unsubscribe, honest subject lines.
- **GDPR** (where applicable) — lawful basis for processing (consent), data
  minimization, right to erasure.
- **AWS SES Acceptable Use Policy** — all recipients have opted in, we process
  bounces and complaints, and we maintain low complaint rates.

Please let us know if any further information is needed. Thank you for your
time and review.

Best regards,
{{your_name}}
{{your_email}}
{{your_aws_account_id}}

---

# Part B — Follow-Up Reply (after DENIED or when stuck PENDING)

**Use when:** your request was denied or is pending, and you need to reply to
the existing case with additional detail. Reply **within the same case
thread** — do not open a new case.

**Subject:** Re: Case {{case_id}} — Production Access Request — Additional Information

**Body:**

Hello AWS Support,

Thank you for reviewing our production access request (Case {{case_id}}).
I'd like to provide additional detail about our use case and sending practices.

## Mail Type: {{mail_type}}

We selected **{{mail_type}}** because our emails are
{{justification_for_mail_type}}.

*(Example justifications)*
- For MARKETING: "…a weekly newsletter sent to subscribers who have explicitly
  opted in to receive updates about our product."
- For TRANSACTIONAL: "…system-generated password-reset emails and order
  confirmations triggered by user actions on our platform."

## Use Case & Website

Our website **{{website_url}}** {{description_of_what_the_site_does_and_how_users_sign_up}}.

We send the following types of email:
1. {{email_type_1_description}}
2. {{email_type_2_description}}

## Expected Volume

We expect to send approximately **{{expected_volume}}** emails per
{{day|week|month}}, growing gradually as our user base expands. Our initial
volume is well within the default production limit of 50,000 emails per 24 hours.

## Email Address Collection & Consent (Opt-In)

Every recipient on our list has provided **explicit consent** through the
following mechanism:

- **Opt-in method:** {{opt_in_mechanism}}
  *(Example: "Users subscribe via a sign-up form on our website. They receive
  a confirmation email (double opt-in) and must click the verification link
  before any further email is sent. We do not purchase, scrape, or rent email
  lists.")*
- **Consent records:** We maintain timestamps and IP addresses of opt-in
  confirmations in our database for audit purposes.

## Bounce Handling

We process hard bounces automatically:

- SES sends bounce notifications to an **SNS topic** configured on our verified
  domain identity.
- Our application subscribes to this SNS topic and **automatically suppresses**
  any address that generates a hard bounce.
- Bounced addresses are flagged in our database and excluded from all future
  sends.
- Soft bounces are tracked; addresses that soft-bounce more than
  {{soft_bounce_threshold}} consecutive times are also suppressed.

## Complaint Handling

We take spam complaints seriously and process them immediately:

- SES sends complaint notifications to the **same SNS topic** used for bounces.
- Our application **immediately and permanently suppresses** any address that
  generates a complaint.
- Complainants are removed from all lists automatically — no manual intervention
  required.
- We monitor our complaint rate through the SES reputation dashboard and keep it
  below 0.1% (well under AWS's acceptable threshold).

## Unsubscribe / Opt-Out

Every marketing email we send includes:

- A **one-click unsubscribe link** in the footer that instantly removes the
  recipient from all lists.
- A **physical mailing address** (as required by CAN-SPAM).
- Clear sender identification (our brand name and verified domain).

Unsubscribe requests are processed **immediately** and take effect for all
future sends. Recipients who unsubscribe are never re-added without a new,
explicit opt-in.

## Compliance

We comply with:
- **CAN-SPAM Act** — physical address, clear sender identity, working
  unsubscribe, honest subject lines.
- **GDPR** (where applicable) — lawful basis for processing (consent), data
  minimization, right to erasure.
- **AWS SES Acceptable Use Policy** — all recipients have opted in, we process
  bounces and complaints, and we maintain low complaint rates.

---

We believe our sending practices meet AWS's requirements for production access.
Please let us know if any further information is needed. Thank you for your
time and review.

Best regards,
{{your_name}}
{{your_email}}
{{your_aws_account_id}}
