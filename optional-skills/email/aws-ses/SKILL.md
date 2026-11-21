---
name: aws-ses
description: "Amazon SES setup: production access, DKIM/SPF, SMTP creds."
version: 0.1.2
author: abebe (airo7), Hermes Agent
license: MIT
dependencies: [boto3]
platforms: [linux, macos]
compatibility: "Any agent that loads SKILL.md (Hermes, Claude Code, Codex, Cursor, OpenCode, OpenClaw). Scripts are plain python3 + boto3."
metadata:
  hermes:
    tags: [AWS, SES, Email, SMTP, DKIM, SPF, Sendy, Newsletter, Marketing]
    related_skills: []
  openclaw:
    requires:
      bins: [python3]
      anyBins: [dig]
      env: [AWS_DEFAULT_REGION]
      optionalEnv: [AWS_PROFILE, SES_DOMAIN]
---

# AWS SES — Sandbox to Production Playbook

Everything you need to go from a fresh AWS account to a working, production-capable
email-sending setup through Amazon SES, including Sendy newsletter integration as a
worked example.

---

## Scope & Permissions

What the bundled scripts actually do, and nothing else:

| Script | AWS calls | Other access | Mutates? |
|---|---|---|---|
| `check_ses_status.py` | `GetAccount`, `GetEmailIdentity` | — | No |
| `bounce_stats.py` | `GetAccount`, `ListSuppressedDestinations`, CloudWatch `GetMetricStatistics` | — | No |
| `verify_dns.py` | `GetEmailIdentity` | runs `dig` (validated names only) | No |
| `setup_domain.py` | `CreateEmailIdentity`, `GetEmailIdentity`, `PutEmailIdentityMailFromAttributes` | — | Yes (`--dry-run` available) |
| `send_test_email.py` | `GetAccount`, `SendEmail` | — | Yes — sends mail (`--dry-run` available) |
| `gen_smtp_password.py` | none (local HMAC) | reads the boto3 credential chain or stdin | Only writes a file if `--output` is given |

Everything else in this document — production-access cases, SNS/configuration
sets, Sendy, `aws` CLI, `curl`, `openssl` snippets — is **reference material
for the user to run themselves**. Agents must:

- ask before running any command that sends email, creates/changes AWS
  resources, or calls a third-party URL (e.g. a Sendy instance);
- never print, log, or paste IAM secrets, SMTP passwords, or API keys;
- not read other skills' directories, install cron jobs or startup hooks, or
  write state outside the user-specified paths.

Network: AWS SES v2 and CloudWatch endpoints in the configured region, plus
DNS lookups via `dig`. No other hosts.

---

## 1. When to Use

Trigger this skill when the user wants to:

- Send email through AWS SES from an application or script.
- Escape the SES sandbox and get production access.
- Verify a domain identity (including DKIM + SPF records).
- Generate SES SMTP credentials from an IAM secret.
- Set up Sendy (self-hosted PHP newsletter app) with SES as the SMTP backend.
- Troubleshoot "email not sending," "domain stuck in PENDING," or
  "production access denied."

---

## 2. Step 1 — Prerequisites & Credentials

### AWS account readiness

You need an AWS account with billing enabled and a **programmatic IAM user** (or role).
Grant only what the current task needs. IAM uses the `ses:` prefix for both the
v1 and v2 SES APIs (there is no `sesv2:` prefix). Replace `<region>`, `<acct>`
and `mydomain.com`.

**Read-only diagnostics** (`check_ses_status.py`, `bounce_stats.py`, `verify_dns.py`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SesReadAccount",
            "Effect": "Allow",
            "Action": [
                "ses:GetAccount",
                "ses:ListSuppressedDestinations",
                "cloudwatch:GetMetricStatistics"
            ],
            "Resource": "*"
        },
        {
            "Sid": "SesReadIdentity",
            "Effect": "Allow",
            "Action": "ses:GetEmailIdentity",
            "Resource": "arn:aws:ses:<region>:<acct>:identity/mydomain.com"
        }
    ]
}
```

**Domain setup** (`setup_domain.py`) — add only while configuring the domain, then remove:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SesDomainSetup",
            "Effect": "Allow",
            "Action": [
                "ses:CreateEmailIdentity",
                "ses:GetEmailIdentity",
                "ses:PutEmailIdentityMailFromAttributes"
            ],
            "Resource": "arn:aws:ses:<region>:<acct>:identity/mydomain.com"
        }
    ]
}
```

**Sending** (`send_test_email.py`, SMTP users) — locked to one identity and one
From domain:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SesSendFromMyDomain",
            "Effect": "Allow",
            "Action": ["ses:SendEmail", "ses:SendRawEmail"],
            "Resource": "arn:aws:ses:<region>:<acct>:identity/mydomain.com",
            "Condition": {
                "StringLike": {"ses:FromAddress": "*@mydomain.com"}
            }
        }
    ]
}
```

If you send with a configuration set, also allow its ARN
(`arn:aws:ses:<region>:<acct>:configuration-set/<name>`) on the sending
statement. Use a **separate** IAM user for SMTP credentials (Sendy, your app)
with only the sending policy — its secret becomes the SMTP password.

### Credentials

Use the standard AWS credential chain — never put keys in shell commands, in
this skill's folder, or in a repository:

```bash
aws configure --profile ses        # or: aws configure sso --profile ses
export AWS_PROFILE=ses
export AWS_DEFAULT_REGION=eu-north-1   # or nearest SES region
```

On EC2/ECS/Lambda, attach an IAM role instead and set only the region.

If you must use a `.env` file, keep it **outside** any git repo and outside the
installed skill directory, and `chmod 600` it. This repo's `.gitignore` and
`.skillignore` exclude `.env*`, `*.pem` and `credentials` files as a backstop.

### Region choice

SES is available in most commercial AWS regions. Pick the one closest to your users:

| Region | Code |
|---|---|
| US East (N. Virginia) | us-east-1 |
| US West (Oregon) | us-west-2 |
| EU (Stockholm) | eu-north-1 |
| EU (Ireland) | eu-west-1 |
| EU (Frankfurt) | eu-central-1 |
| Asia Pacific (Sydney) | ap-southeast-2 |
| Asia Pacific (Tokyo) | ap-northeast-1 |

If no strong reason to choose otherwise, **eu-north-1** is a safe default for EU users.

---

## 3. Step 2 — Check Account Status (Sandbox vs Production)

Run the status script:

```bash
export SES_DOMAIN=mydomain.com   # optional; script also accepts a CLI argument
python3 scripts/check_ses_status.py mydomain.com
```

### Output fields explained

| Field | Meaning |
|---|---|
| `production_access` | `true` after AWS approves your production request. Until then you are in the **sandbox**. |
| `sending_enabled` | `true` means your account is not under a sending suspension. |
| `enforcement` | Should read `HEALTHY`. Anything else (e.g. `GRACEFUL`, `PROBATION`) means you have delivery issues. |
| `quota.Max24HourSend` | How many emails you may send per 24-hour window. |
| `quota.MaxSendRate` | Maximum sending rate in emails per second. |
| `quota.SentLast24Hours` | How many emails you have already sent today. |
| `review.Status` | `PENDING`, `GRANTED`, or `DENIED`. |
| `review.CaseId` | AWS Support case ID for your production access case. |
| `mail_type` | The mail type recorded in your support case. |
| `website` | The website URL recorded in your support case. |
| `domain_verification` | `SUCCESS` / `PENDING` / `FAILED` for the domain identity. |
| `dkim` | DKIM signing attributes for the domain. |

### Sandbox vs Production limits

| Limit | Sandbox | Production |
|---|---|---|
| Max 24h sends | 200 | 50,000 (default; auto-ramps with good reputation) |
| Max send rate | 1 / sec | 14 / sec (default; ramps) |
| Recipients | Verified addresses only | Any address |
| Outbound to unverified | Blocked | Allowed |

**If `production_access` is `false` and `sending_enabled` is `true`, you're in sandbox.**
Proceed to domain verification (Step 3), then production access (Step 4).

---

## 4. Step 3 — Domain Verification + DKIM + SPF

### 4a. Verify the domain in SES

Create the domain identity in the SES console or via CLI:

```bash
aws sesv2 create-email-identity --email-identity mydomain.com
```

Or use the AWS Console: SES → Verified identities → Create identity → Domain.

### 4b. DNS records to add

You will get **three DKIM CNAME records** and you should also add an **SPF TXT record**.

#### DKIM (`<selector>` values given by AWS — they look like `abcdefg...`):

```
Name:  <selector1>._domainkey.mydomain.com
Type:  CNAME
Value: <selector1>.dkim.amazonses.com.

Name:  <selector2>._domainkey.mydomain.com
Type:  CNAME
Value: <selector2>.dkim.amazonses.com.

Name:  <selector3>._domainkey.mydomain.com
Type:  CNAME
Value: <selector3>.dkim.amazonses.com.
```

**Important:** The trailing dots (`.`) on the value side are part of the CNAME target.
If your DNS provider appends the zone name automatically (common), check what you
actually entered — a missing trailing dot can result in a double-zone suffix like
`<selector>.dkim.amazonses.com.mydomain.com`.

#### SPF TXT record:

```
Name:  mydomain.com   (or "@" depending on DNS provider)
Type:  TXT
Value: "v=spf1 +a +mx include:amazonses.com ~all"
```

If you already have an SPF record, add `include:amazonses.com` before the final `~all` or `-all`:

```
"v=spf1 +a +mx include:_spf.google.com include:amazonses.com ~all"
```

### 4c. Verify with the DNS check script

```bash
python3 scripts/verify_dns.py mydomain.com
```

It runs `dig +short` for all three DKIM selectors and the SPF record, printing PASS/FAIL.

### 4d. Troubleshooting DNS

**"Record points to a target inside this zone that does not exist" (Hetzner DNS Console):**
This is a false-positive warning from some DNS providers. They check whether the
CNAME target is itself within the same zone and warn because it isn't. The record
is actually correct. Confirm with `dig`:

```bash
dig +short <selector>._domainkey.mydomain.com CNAME
# Should return: <selector>.dkim.amazonses.com.
```

If it returns the correct target, the record is fine regardless of the UI warning.

**Domain stuck in PENDING:**
- DNS propagation can take up to 72 hours, but typically 5–30 minutes for major
  providers.
- Cloudflare / Route 53 are usually sub-minute.
- Ensure you didn't accidentally create relative-name CNAMEs (missing trailing dot
  when your provider requires it, or double-zone-suffixing when your provider
  doesn't).
- In the SES console, click "Retry verification" after the DNS has propagated.

**DKIM won't verify even though dig returns the right value:**
- Wait. AWS's DKIM verification poller runs periodically, not instantly.
- Check that the DKIM signing status shows `SUCCESS` in the SES console, not just
  the domain verification status.

---

## 5. Step 4 — Get Production Access (the Critical Playbook)

This is the step where most people stall. Follow this precisely.

### 5a. What AWS Support evaluates

AWS grants production access based on your answers to these questions (asked in the
SES console when you request production access):

1. **Mail type** — TRANSACTIONAL vs MARKETING vs other.
2. **Website URL** — where you collect recipient addresses.
3. **Use-case description** — what emails you send and why.
4. **Opt-in / consent mechanism** — how recipients agree to receive mail.
5. **Bounce and complaint handling** — what happens when delivery fails or a
   recipient reports spam.
6. **Unsubscribe mechanism** — how recipients opt out.

### 5b. THE #1 REJECTION REASON: Wrong Mail Type

**This is the single most common reason for denial.** AWS is strict about
classification:

| If you send… | Choose… |
|---|---|
| Password resets, order confirmations, receipts, one-time codes, account notifications | **TRANSACTIONAL** |
| Newsletters, marketing campaigns, product updates, drip sequences, weekly digests | **MARKETING** |

**Crucially: if you classify a newsletter/marketing use case as TRANSACTIONAL, your
request WILL be denied.** AWS expects newsletters to go through MARKETING because
MARKETING requires stronger opt-in, bounce, and unsubscribe handling — and AWS wants
to see that you understand those requirements.

**When in doubt:** be honest. If your email has any promotional or newsletter
character, use MARKETING. Transactional is only for system-triggered, one-to-one,
non-promotional messages.

### 5c. Writing the support case (and the reply template)

Request production access through the SES console: Account dashboard →
"Request production access."

Fill in the form carefully. If denied, you get a **support case ID**. The key insight:

> **You cannot resubmit via the API or console while status is DENIED.**
> `ConflictException` is thrown. You MUST reply to the **existing support case**
> — do NOT open a new case.

Use `templates/support-case-reply.md` as a template. It contains **two parts**:

- **Part A — Initial Request:** open the production-access case with this
  content (covers mail type, use case, volume, opt-in, bounce/complaint
  handling, unsubscribe, compliance — everything AWS evaluates up front).
- **Part B — Follow-Up Reply:** reply to the **existing** case with this if
  DENIED or stuck PENDING (never open a new case while status is DENIED).

Fill in the placeholders with your real details:

- `{{website_url}}` — where users sign up (e.g. `https://mysaas.com/newsletter`)
- `{{verified_domain}}` — the domain identity you verified (e.g. `yourdomain.com`)
- `{{expected_volume}}` — e.g. `6,000 emails per week (one weekly newsletter to ~6K subscribers)`
- `{{mail_type}}` — `MARKETING` or `TRANSACTIONAL` (see table above)
- `{{justification_for_mail_type}}` — one sentence why that type fits
- `{{description_of_what_the_site_does_and_how_users_sign_up}}` — site purpose + signup flow
- `{{email_type_1_description}}` / `{{email_type_2_description}}` — what you send
- `{{opt_in_mechanism}}` — how consent is obtained (e.g. `Double opt-in via email confirmation link after signup form submission`)
- `{{soft_bounce_threshold}}` — e.g. `3`
- `{{case_id}}` — the AWS support case ID (Part B only)
- `{{your_name}}` / `{{your_email}}` / `{{your_aws_account_id}}` — sign-off

### 5d. What a strong reply looks like

A strong case-reply:

1. **Clearly states the mail type** with a justification.
2. **Describes the exact opt-in flow** — "double opt-in" carries weight.
3. **Names the AWS services** used for bounce/complaint handling (SNS, Lambda).
4. **Explains how you suppress** hard bounces and complaints programmatically
   (not "we'll look at the dashboard sometimes").
5. **Mentions CAN-SPAM/GDPR compliance** — footer links, physical address,
   one-click unsubscribe.

Weak replies get rejected again. Be specific.

### 5e. After approval

Once your case status changes to GRANTED and `ProductionAccessEnabled` is `true`:

- It may take 15–30 minutes for the change to propagate.
- Your sending limits start at the default production tier (~50K/day, 14/sec)
  and auto-ramp as you maintain good sending reputation.
- Run `check_ses_status.py` again to confirm.

---

## 6. Step 5 — Generate SMTP Credentials

SES SMTP credentials are **not** your IAM secret key. They are derived from it
locally with a SigV4-style HMAC-SHA256 chain (see below). Nothing is sent to AWS.

### Run the derivation script

```bash
# Recommended: write to a new 0600 file outside any repo, then load it into your app's secret store.
python3 scripts/gen_smtp_password.py --region eu-north-1 --output ~/ses-smtp.env
```

The secret comes from the boto3 credential chain (`AWS_PROFILE`, shared
credentials file, SSO, or an instance role). There is **no** `--secret` flag —
command-line arguments are visible in `ps` and shell history. To derive for a
different key, pipe it in: `python3 scripts/gen_smtp_password.py --region eu-north-1 --secret-stdin --output ~/ses-smtp.env < keyfile`.

Without `--output`, the script prints only the non-secret fields and hides the
password. `--show-password` prints it to stdout — avoid it in agent sessions,
CI logs, or screen shares.

```
SMTP_USERNAME: AKIA...
SMTP_PASSWORD: <hidden — rerun with --output PATH or --show-password>
SMTP_SERVER:   email-smtp.eu-north-1.amazonaws.com
SMTP_PORT:     587
```

### How the derivation works

The SMTP password is a SigV4-style derivation, not the raw IAM secret:

1. Sign the literal `"11111111"` with HMAC-SHA256, keyed by `"AWS4" + secret`.
2. Chain the signature through the region, then `"ses"`, then `"aws4_request"`.
3. Sign the literal `"SendRawEmail"` with the resulting key.
4. Base64-encode a `0x04` version byte prepended to that signature.

The password is **region-dependent** — the same secret yields a different
password per region, so `--region` is required. The username is the IAM
Access Key ID (same key pair as the secret). This is the standard AWS
algorithm documented for SES SMTP credentials.

### SMTP endpoints

| Port | Protocol | Notes |
|---|---|---|
| 587 | STARTTLS | **Recommended.** Use this. |
| 2587 | STARTTLS | Alternative for networks that block 587. |
| 465 | TLS Wrapper | Legacy; works but 587 is preferred. |
| 25 | Plain / STARTTLS | Often blocked by cloud providers. |


```bash
# Test SMTP connectivity (openssl)
openssl s_client -connect email-smtp.eu-north-1.amazonaws.com:587 -starttls smtp -crlf
# EHLO mydomain.com
# AUTH LOGIN
# (paste base64(username))
# (paste base64(password))
```

### Rotating credentials

If you need to rotate, create a new IAM access key pair, derive the SMTP password
from the new secret, update your app config, and then deactivate/delete the old key.

---

## 7. Step 6 — Send a Test Email

### Important: Sandbox restriction

In sandbox, you can ONLY send to verified email addresses or domains. Both the
FROM and TO addresses must be verified.

### Minimal Python send (boto3 sesv2)

```python
import boto3
import os

client = boto3.client("sesv2", region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1"))

response = client.send_email(
    FromEmailAddress="hello@mydomain.com",
    Destination={
        "ToAddresses": ["hello@mydomain.com"],  # sandbox: must be verified
    },
    Content={
        "Simple": {
            "Subject": {"Data": "SES test email", "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": "This is a test email from Amazon SES.", "Charset": "UTF-8"}
            },
        }
    },
)

print(f"MessageId: {response['MessageId']}")
```

### Verify receipt

- Check the inbox in your email client.
- SES events (delivery status) appear in the SNS topic if you configured one.
- In SES console: go to "Email sending" → "Search for emails" and filter by
  MessageId.

---

## 8. Step 7 — Sendy Integration (Worked Example)

[Sendy](https://sendy.co) is a self-hosted PHP newsletter application that uses
SES as its SMTP backend. This section assumes you have Sendy installed on a server
and want to configure it to send through your freshly provisioned SES setup.

### 8a. Sendy SMTP configuration

In Sendy's settings (`/includes/config.php` or via the web admin):

```php
// SMTP settings
define('SMTP_HOST', 'email-smtp.eu-north-1.amazonaws.com');  // your SES region's endpoint
define('SMTP_PORT', 587);
define('SMTP_AUTH', true);
define('SMTP_SECURE', 'tls');
define('SMTP_USERNAME', 'AKIA...');        // IAM Access Key ID
define('SMTP_PASSWORD', 'BLBM...');       // derived SMTP password (not IAM secret!)
define('SMTP_FROM_ADDRESS', 'hello@mydomain.com');
define('SMTP_FROM_NAME', 'My Newsletter');
```

**Common Sendy SES pitfalls:**

- Using the raw IAM secret as the SMTP password instead of the derived password.
- Using port 465 with TLS wrapper instead of port 587 with STARTTLS.
- FROM address not verified in SES (in sandbox) or not matching the verified domain.

### 8b. Mail type for Sendy

**A Sendy newsletter is MARKETING mail type.** Even if the content is
"transactional" in your mind (e.g., a product update), AWS SES classifies
bulk/app-mass sends as MARKETING. Classify correctly or your production access
request will be denied.

### 8c. Sendy API examples

Sendy exposes a simple HTTP API for managing campaigns and subscribers.

> **For the user to run, not the agent.** These calls go to *your own* Sendy
> server and can email real subscribers. An agent must not run them without
> explicit confirmation of the URL, the list, and whether to send. Keep the API
> key in an environment variable — never inline it.

```bash
export SENDY_URL=https://sendy.yourdomain.com
read -rs SENDY_API_KEY && export SENDY_API_KEY   # prompt, not shell history
```

#### Create a campaign (draft or send immediately):

```bash
curl -X POST "$SENDY_URL/api/campaigns/create.php" \
  --data-urlencode "api_key=$SENDY_API_KEY" \
  -d "brand_id=1" \
  -d "list_ids=1,2" \
  -d "title=Weekly Digest" \
  -d "subject=Your Weekly Update" \
  -d "from_name=My Newsletter" \
  -d "from_email=hello@mydomain.com" \
  -d "reply_to=hello@mydomain.com" \
  -d "plain_text=Fallback plain text version" \
  -d "html_text=<h1>Hello!</h1><p>This is the newsletter.</p>" \
  -d "send_campaign=0"            # 0 = draft (review in Sendy first), 1 = send immediately
```

#### Subscribe an email address:

```bash
curl -X POST "$SENDY_URL/subscribe" \
  --data-urlencode "api_key=$SENDY_API_KEY" \
  -d "list=LIST_ID" \
  -d "email=subscriber@example.com" \
  -d "name=John Doe" \
  -d "boolean=true"              # true to send confirmation email (double opt-in)
```

#### Subscriber management endpoints:

| Endpoint | Purpose |
|---|---|
| `/subscribe` | Add subscriber (with optional double opt-in) |
| `/unsubscribe` | Remove subscriber |
| `/api/subscribers/active-subscriber-count.php` | Get list count |
| `/api/subscribers/subscription-status.php` | Check if email is subscribed |

### 8d. Production volume example

A typical Sendy setup with 6,000 subscribers and one weekly newsletter:
- **6,000 emails per week**
- **~25,000 emails per month**
- This is well within default production limits (50K/day).
- Your Max24HourSend will auto-ramp from 50K upward as your reputation improves.

In your AWS support case, you'd state: `MARKETING, 6,000 emails/week, weekly
newsletter to opt-in subscribers`.

---

## 9. SNS Bounce & Complaint Handling

SES sends delivery, bounce, and complaint notifications via SNS (Simple
Notification Service). You MUST wire these up — AWS will suspend your
account if your complaint rate exceeds 0.1% (and even 0.08% should trigger
action). Bounce rate matters too: **>5% triggers a review, >10% triggers
suspension** — these are the more common trigger. **"We check the dashboard"
is not a strategy; it's why production access gets denied.**

### Architecture

```
SES send → Delivery/Bounce/Complaint event
              ↓
         SNS Topic (e.g. ses-events)
              ↓
         SQS / Lambda / HTTP endpoint (subscriber)
              ↓
         Your app: suppress bounces + complaints, log deliveries
```

### Setting it up

1. **Create an SNS topic** (standard type, not FIFO):
   ```bash
   aws sns create-topic --name ses-events
   ```

2. **Configure the SES configuration set** to publish to that topic. A
   configuration set is the bridge between SES sends and SNS:
   ```bash
   aws sesv2 create-configuration-set --configuration-set-name my-events
   aws sesv2 create-configuration-set-event-destination \
       --configuration-set-name my-events \
       --event-destination-name all-events \
       --event-destination '{
           "SnsDestination": {"TopicArn": "arn:aws:sns:<region>:<acct>:ses-events"},
           "MatchingEventTypes": ["SEND", "REJECT", "BOUNCE", "COMPLAINT",
                                  "DELIVERY", "OPEN", "CLICK"]
       }'
   ```

3. **Include the configuration set** when sending:
   ```python
   client.send_email(
       ...
       ConfigurationSetName="my-events",
   )
   ```

   Or set a **default configuration set** on the email identity (simpler —
   all sends from that domain automatically use it):
   ```bash
   aws sesv2 put-email-identity-configuration-set-attributes \
       --email-identity mydomain.com \
       --configuration-set-name my-events
   ```

4. **Subscribe a Lambda or HTTP endpoint** to the SNS topic:
   ```bash
   aws sns subscribe \
       --topic-arn arn:aws:sns:<region>:<acct>:ses-events \
       --protocol lambda \
       --notification-endpoint arn:aws:lambda:<region>:<acct>:function:ses-handler
   ```

### What your handler must do

| Event type | Action |
|---|---|
| **BOUNCE** (hard) | Suppress immediately. Hard bounces mean permanent failure (mailbox doesn't exist, domain doesn't exist). |
| **BOUNCE** (soft) | Track count. Suppress after N consecutive soft bounces (3 is a good default). Soft bounces are transient (mailbox full, server busy). |
| **COMPLAINT** | Suppress immediately and permanently. Someone clicked "Report spam." Never email this address again. |
| **DELIVERY** | Optional: log for analytics, reset soft-bounce counters. |

A minimal Lambda handler (Python):

```python
import json

def handler(event, context):
    for record in event.get("Records", []):
        msg = json.loads(record["Sns"]["Message"])
        ntype = msg.get("notificationType", "")

        if ntype == "Bounce":
            for recipient in msg.get("bounce", {}).get("bouncedRecipients", []):
                email = recipient.get("emailAddress")
                htype = recipient.get("bounceType")  # Permanent vs Transient
                if htype == "Permanent":
                    suppress(email, reason="hard-bounce")
                else:
                    track_soft_bounce(email)

        elif ntype == "Complaint":
            for recipient in msg.get("complaint", {}).get("complainedRecipients", []):
                suppress(recipient.get("emailAddress"), reason="complaint")

    return {"statusCode": 200}
```

### SNS notification format

The SNS message body is a **JSON string** (double-encoded). Parse it as:

```python
msg = json.loads(record["Sns"]["Message"])
# msg["notificationType"] → "Bounce" | "Complaint" | "Delivery"
# msg["bounce"]["bouncedRecipients"][0]["emailAddress"]
# msg["complaint"]["complainedRecipients"][0]["emailAddress"]
```

### SES sending without a configuration set

If you don't specify a configuration set, SES still generates events for
bounces and complaints — but they go to the account-level **notification
settings**, not to your topic. Configure these as a fallback:

```bash
aws ses set-identity-notification-topic \
    --identity mydomain.com \
    --notification-type Bounce \
    --sns-topic arn:aws:sns:<region>:<acct>:ses-events

aws ses set-identity-notification-topic \
    --identity mydomain.com \
    --notification-type Complaint \
    --sns-topic arn:aws:sns:<region>:<acct>:ses-events
```

**Best practice:** Use a configuration set for new sends and set the
account-level notification as a safety net for anything that slips through.

---

## 10. Newsletter Sending Strategy (Warmup, Rate Limits, List Hygiene)

If you are sending newsletters or marketing campaigns through SES, a
disciplined approach is the difference between a healthy sender reputation
and an account under PROBATION.

### 10a. IP warmup — the first 2-4 weeks

AWS starts new SES accounts on **shared IP pools** by default. Shared IPs
have established reputation, so your warmup period is shorter than with
dedicated IPs. But you still need to ramp gradually:

| Week | % of max list size | Notes |
|---|---|---|
| 1 | 5-10% | Send to your most engaged subscribers first. |
| 2 | 15-25% | Add moderately-engaged addresses. |
| 3 | 30-50% | Broaden to less-engaged, but stop if complaint rate rises. |
| 4 | 70-100% | Full list, assuming complaint rate holds steady. |

**SES auto-ramps** your sending quota based on reputation. If you send
high-quality mail (low bounce, low complaint), your Max24HourSend goes
from 50K → 100K → 200K+ automatically. If you trigger complaints, it
stays flat or drops.

### 10b. List hygiene rules

1. **Send only to people who have explicitly opted in within the last
   12 months.** Older addresses are stale — people change jobs, abandon
   accounts, forget they signed up. The single biggest factor in spam
   complaints is "I don't remember signing up for this."

2. **Remove hard bounces within one send cycle.** A bounced address
   should never receive another email. Boto3 `sesv2` has a suppression
   list API — use it or maintain your own. If you hit the same bad
   address repeatedly, mailbox providers (Gmail, Yahoo, Outlook) will
   treat your domain as a spam source.

3. **Purge addresses that haven't engaged in 6+ months** (no opens, no
   clicks). Inactive addresses become spam traps over time — mailbox
   providers convert abandoned accounts into honeypots to catch senders
   who don't clean their lists. Sending to a spam trap is instant
   reputation damage. Run a re-engagement campaign before purging:
   send a "We've missed you — click to stay subscribed" email. If no
   engagement after 2 sends, delete.

4. **Monitor complaint rate obsessively.** Run `bounce_stats.py` daily:
   ```bash
   python3 scripts/bounce_stats.py
   ```
   If the complaint rate crosses 0.05%, investigate immediately. Above
   0.1%, AWS may place your account under review.

5. **Use double opt-in.** Confirmed opt-in (send a verification link,
   require a click before any further email) eliminates typos, bot
   signups, and malicious subscription attempts. It adds friction but
   produces a vastly cleaner list.

### 10c. Rate limits & throttling

SES enforces a per-account **MaxSendRate** (emails/second). In sandbox
it's 1/sec; in production it starts at ~14/sec and ramps. If you exceed
it, SES v2 raises `TooManyRequestsException` — your sending code must
handle this with **exponential backoff**. (botocore's default retry mode
handles this automatically in most SDKs; the manual loop below is for
clients that don't.)

```python
import time
import boto3
from botocore.exceptions import ClientError

def send_with_retry(client, max_retries=5, **kwargs):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return client.send_email(**kwargs)
        except ClientError as e:
            if e.response["Error"]["Code"] == "TooManyRequestsException":
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2   # 1s → 2s → 4s → 8s → 16s
            else:
                raise
```

For bulk sends, **pace yourself** — don't blast your entire list at 14/sec
in one loop. Spread sends over time using a queue (SQS) or by sleeping
between batches.

### 10d. Dedicated IPs

Request dedicated IPs through an AWS Support case when you send **>500K
emails/month consistently**. Dedicated IPs give you full control over
reputation but require deliberate warmup (start at 50 emails/day per IP,
double weekly). For most users, shared IPs are better — AWS manages the
reputation so you don't have to.

---

## 11. Troubleshooting Common Issues

### 11a. Emails landing in spam

**Checklist:**

1. **DKIM + SPF must PASS.** Run `verify_dns.py`. If either fails, email
   providers cannot verify that you are authorized to send from that
   domain. No DKIM + no SPF = spam folder essentially guaranteed.
   ```bash
   python3 scripts/verify_dns.py mydomain.com
   ```

2. **DMARC record.** Add a DMARC policy at `_dmarc.mydomain.com`:
   ```
   Type:  TXT
   Name:  _dmarc.mydomain.com
   Value: "v=DMARC1; p=none; rua=mailto:dmarc@mydomain.com"
   ```
   Start with `p=none` (monitor), move to `p=quarantine`, then
   `p=reject` once you trust your setup.

3. **Custom MAIL FROM domain.** Without it, SES uses a default
   `amazonses.com` MAIL FROM, which causes SPF alignment to fail
   (DMARC depends on it). Set a custom MAIL FROM:
   ```bash
   aws sesv2 put-email-identity-mail-from-attributes \
       --email-identity mydomain.com \
       --mail-from-domain bounce.mydomain.com
   ```
   Then add an MX record for `bounce.mydomain.com` (for inbound bounce
   processing) and an SPF TXT record.

4. **Content signals.** Avoid ALL CAPS subject lines, excessive
   exclamation marks, image-only emails with no text, URL shorteners,
   and "click here" as your only link text. These are spam-trigger
   signals for Bayesian filters.

5. **Warm up the domain.** A brand-new domain sending to 10K addresses
   on day 1 looks like a spam run. Ramp over 2-4 weeks (see §10a).

6. **Check blacklists.** Use `mxtoolbox.com` or `multirbl.valli.org` to
   check if your IP/domain is on any DNS blacklists.

### 11b. DKIM verification stuck in PENDING

- **DNS hasn't propagated**: Wait. Up to 72 hours for some providers,
  though Cloudflare/Route 53 are typically sub-minute. Check with `dig`:
  ```bash
  dig +short <token>._domainkey.mydomain.com CNAME
  ```
  If `dig` returns the right target, your DNS is fine — SES's periodic
  verification scanner hasn't picked it up yet. **Manually trigger
  re-verification** from the SES console (Verified identities → select
  domain → "Retry" or "Verify" button). This forces an immediate check
  rather than waiting for the automated poller.

- **CNAME record has wrong target**: Compare with the SES console. The
  three DKIM tokens are unique to your domain/region. If you re-created
  the identity, old CNAME records may point to stale tokens.

- **Trailing-dot issues** (Hetzner, some registrars): See §4d — verify
  with `dig +short`, not the DNS provider's UI.

- **Identity created in the wrong region**: SES identities are regional.
  If you created the identity in `eu-north-1` but are checking in
  `us-east-1`, it won't exist. Always specify `--region`.

### 11c. 454 TooManyRequestsException

**"TooManyRequestsException: Maximum sending rate exceeded."** SES is
rate-limiting you. Actions:

1. **Immediate**: add exponential backoff (see §10c).
2. **Check current limits**:
   ```bash
   python3 scripts/check_ses_status.py
   ```
   Look at `MaxSendRate`. If you need more throughput than the default
   (14/sec in production), you can request a limit increase through
   the AWS SES console (Sending Statistics → Request increase).
   Increases are typically granted if your account has good reputation.

3. **Pacing**: don't spin-loop all sends at MaxSendRate. Use a queue or
   sleep between batches.

### 11d. Sending suspended / account under review

If `EnforcementStatus` is `GRACEFUL` or `PROBATION` instead of
`HEALTHY`, your account has delivery issues. This means:

- Your bounce rate or complaint rate crossed AWS's thresholds.
- AWS is monitoring your sends more closely.
- If it degrades to a full suspension, you cannot send at all.

**Recovery:**
1. Stop all sends immediately.
2. Run `bounce_stats.py` to diagnose.
3. Remove every address that has bounced or complained.
4. Audit your subscription flow — are people genuinely opting in?
5. Implement double opt-in if you haven't already.
6. Reply to the AWS support case and explain your remediation steps.

**Prevention:** Wire up SNS bounce/complaint handling (§9) before you
ever send a newsletter. Real-time suppression prevents reputation damage.

---

## 12. Cost Estimation

SES pricing is simple and cheap compared to dedicated email services.

### Sending costs

| Component | Price | Notes |
|---|---|---|
| **Outbound email** | $0.10 / 1,000 emails | Standard rate. |
| **Attachments** | $0.12 / GB | Charged per GB of attachment data, not just the attachment size — encoding adds ~33% overhead. A 1 MB PDF attached to 1,000 emails = ~1 GB = $0.12. |
| **Inbound email** | $0.10 / 1,000 received | If you use SES for receiving mail too. |
| **EC2 to SES** | $0.10 / 1,000 emails | AWS retired the 62,000/month free tier in Aug 2023; replaced with a 3,000 messages/month charge-back for 12 months. Re-check current pricing. |
| **Lightsail to SES** | Free | Emails sent from Lightsail instances: first 1,000/month free. |

### Example cost scenarios

| Scenario | Monthly cost |
|---|---|
| 5,000 emails/month (small SaaS, transactional) | $0.50 |
| 25,000 emails/month (weekly newsletter, 6K subs) | $2.50 |
| 100,000 emails/month (daily marketing, 5K subs) | $10.00 |
| 1,000,000 emails/month (large newsletter) | $100.00 |
| 10,000,000 emails/month (enterprise) | $1,000.00 |

### What SES does NOT charge for

- Domain verification
- DKIM signing
- SMTP credential generation
- SNS topics (standard tier — SNS charges separately for notifications
  above the free tier: 1M publishes/month free, then $0.50/M)

### Comparison with competitors

| Service | Price / 1,000 | Notes |
|---|---|---|
| **AWS SES** | $0.10 | Cheapest by far. Requires your own integration. |
| SendGrid (Twilio) | $0.54–$0.82 | Managed UI + API. Free tier: 100/day. |
| Mailgun | $0.40–$0.80 | Developer-friendly API. Free tier: 100/day. |
| Postmark | $1.25+ | Transactional only, no marketing. |
| Mailchimp (Mandrill) | ~$0.40 | Marketing focused. |
| ConvertKit | $29+/month flat | For creators; includes full CRM. |

**For self-hosted apps and newsletters (Sendy + SES):** SES is the winner
on cost. You pay ~$0.10/1,000 for sending vs $30-100+/month for the
equivalent on a managed platform.

---

## 13. Common Pitfalls

1. **Marketing classified as TRANSACTIONAL → instant denial.**
   Newsletters, campaigns, drip sequences, product updates = MARKETING.
   Only system-triggered personal messages = TRANSACTIONAL.

2. **API resubmit after DENIED is blocked.**
   You'll get `ConflictException`. Reply to the existing support case instead.
   Opening a new case will be closed as duplicate.

3. **SMTP password ≠ IAM secret.**
   The SMTP password is derived via HMAC-SHA256. Using the raw IAM secret as
   the SMTP password will fail authentication. Use `gen_smtp_password.py`.

4. **Sandbox: only verified recipients.**
   In sandbox you can only send TO verified email addresses/domains. Trying to
   send to an unverified address returns `MessageRejected: Email address is not
   verified`.

5. **DKIM trailing-dot / relative-name issues in Hetzner and similar DNS
   providers.**
   The DNS console may warn about "target inside this zone" or silently append
   the zone name to CNAME values. Always verify with `dig +short` from the CLI,
   not the DNS provider's UI.

6. **SPF propagation lag after adding `include:amazonses.com`.**
   SPF TXT records can take longer to propagate than CNAME records on some
   providers. If DKIM is verified but SPF checks still fail, wait 15–30 minutes
   and re-check.

7. **Never echo keys in shell commands.**
   `AWS_SECRET_ACCESS_KEY` and derived SMTP passwords should never appear in
   shell history or log output. Use an `AWS_PROFILE` or IAM role; write SMTP
   credentials with `gen_smtp_password.py --output` rather than printing them.

---

## 14. Verification Checklist

After completing all steps, confirm every item:

- [ ] Domain identity status: **SUCCESS** (SES Console → Verified Identities)
- [ ] DKIM signing status: **SUCCESS** (all three CNAMEs verified)
- [ ] DKIM `VerifiedForSendingStatus`: **true** (check via `check_ses_status.py`)
- [ ] SPF record includes `include:amazonses.com` (check via `verify_dns.py`)
- [ ] DMARC record published at `_dmarc.mydomain.com` (start with `p=none`)
- [ ] Custom MAIL FROM domain configured (`setup_domain.py --mail-from`)
- [ ] Production access: **ProductionAccessEnabled = true**
- [ ] Account enforcement: **HEALTHY** (not GRACEFUL or PROBATION)
- [ ] Sending enabled: **true**
- [ ] SMTP credentials generated and tested (port 587, STARTTLS)
- [ ] Test email sent and **received** in inbox (check spam folder too)
- [ ] Test email also sent via `send_test_email.py --dry-run` first, then live
- [ ] SNS bounce/complaint topic created and subscribed (Lambda or HTTP)
- [ ] Configuration set created and attached to the email identity
- [ ] Bounce/complaint suppression automated (not manual dashboard checks)
- [ ] Bounce/complaint counts monitored (run `bounce_stats.py` weekly)
- [ ] Sendy (or your app) configured with the **derived** SMTP password
- [ ] Unsubscribe link present in every email footer
- [ ] Email footer includes physical mailing address (CAN-SPAM)
- [ ] List hygiene process in place (purge unengaged, double opt-in active)

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/check_ses_status.py` | Full account health: sandbox/prod, quota, domain verification, DKIM, review status |
| `scripts/gen_smtp_password.py` | Derive SES SMTP password from IAM secret key |
| `scripts/verify_dns.py` | Check DKIM CNAMEs and SPF record via `dig +short` |
| `scripts/send_test_email.py` | Send a test email via SES v2 API with `--dry-run` preview |
| `scripts/setup_domain.py` | Create/verify a domain identity and print DKIM + SPF DNS records |
| `scripts/bounce_stats.py` | Fetch bounce/complaint/suppression list counts and complaint rate |

Run each from the skill root:

```bash
# 1. Check account status
python3 scripts/check_ses_status.py mydomain.com

# 2. Verify DNS records
python3 scripts/verify_dns.py mydomain.com

# 3. Generate SMTP credentials (written to a new 0600 file, not printed)
python3 scripts/gen_smtp_password.py --region eu-north-1 --output ~/ses-smtp.env

# 4. Set up a domain identity + print DNS records
python3 scripts/setup_domain.py mydomain.com --dry-run   # preview first
python3 scripts/setup_domain.py mydomain.com

# 5. Send a test email
python3 scripts/send_test_email.py \
    --from hello@mydomain.com --to me@example.com \
    --subject "Test" --body "Hello from SES" --dry-run   # preview first
python3 scripts/send_test_email.py \
    --from hello@mydomain.com --to me@example.com \
    --subject "Test" --body "Hello from SES"

# 6. Monitor bounce & complaint health (sample addresses masked unless --show-addresses)
python3 scripts/bounce_stats.py
```

All scripts read AWS credentials from the environment (boto3 default credential
chain). No hardcoded keys anywhere.

---

## Further Reading

- [AWS SES Developer Guide — Moving out of the sandbox](https://docs.aws.amazon.com/ses/latest/dg/request-production-access.html)
- [AWS SES SMTP credentials derivation](https://docs.aws.amazon.com/ses/latest/dg/smtp-credentials.html)
- [Sendy — Sending with Amazon SES](https://sendy.co/amazon-ses)
- [AWS SES DKIM setup](https://docs.aws.amazon.com/ses/latest/dg/send-email-authentication-dkim.html)
