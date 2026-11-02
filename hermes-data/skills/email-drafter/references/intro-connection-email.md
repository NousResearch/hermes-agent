# Intro / connection email — NDR links a colleague to an external person (verified 2026-10-21)

Fires on dictation like *"I want to send an email to Gowri marking Sohum on the same email …
introducing him … ask her to get a demonstration and give me feedback … mark Roshni on CC"*, or
more briefly *"just connect the two of you"*, *"put them in the loop"*, *"introduce X to Y"*.

This is NOT an escalation, not a vendor RFQ, and not a reply — it is a new thread whose job is to
hand a live relationship from NDR to a colleague. Draft only, never send.

## 1. Routing — the part that goes wrong silently

- **To** = the colleague who must ACT (here Gowri Singh `gsingh@draas.com`). Not the external person.
- **Cc** = every person NDR named to "mark" **plus** the external person:
  `Sohum Chhabria <sohumachhabria@gmail.com>, Roshni Murjani <rnr@draas.com>`.
- Resolve each colleague's **sending** address from Gmail headers, never the sheet and never a
  `firstname@draas.com` guess:
  `gws_gmail_search(query='from:gsingh@draas.com')` → `Gowri Singh <gsingh@draas.com>` confirm.
  (Voice "Gauri" = **Gowri**; see `contact-email-lookup.md` §2b.)
- External address: take the exact string NDR gave, or the one the person themselves sent on
  WhatsApp. Never reconstruct a `firstname.lastname@gmail.com` pattern.
- **Cc is invisible in draft JSON summaries** — prove it landed by reading the raw MIME (§5).

## 2. Subject

`Introducing <Full Name> (<Company>) - <capability> for <projects>`

Verified: `Introducing Sohum Chhabria (SohumAI) - AI Renders & Interactive 3D for Amber, Oasis, Northstar`

ASCII only (hyphen not em-dash, no `₹`, no `·`) — Gmail's draft renderer breaks on RFC2047 encoded
words in a Subject. The body can carry `&mdash;` / `&middot;` entities freely.

## 3. Body skeleton — five blocks, in this order

1. **Who + how we met.** One line naming the person and the *origin of the introduction* ("introduced
   to me by Kishan at a BNI event"). NDR always credits the introducer; leaving it out is what makes
   his intros feel thin.
2. **What they can actually do.** Concrete numbered capabilities as *demonstrated artefacts*, not
   adjectives:
   - "2D plans converted into highly interactive 3D websites — scroll-driven animations that turn the
     2D plan into 3D as you scroll, and let us walk into the units"
   - "3D interactive models where you select the tower, the floor plan, look at availability, then get
     into the floor and into the house"
   Bad: "does great visuals and is very creative."
3. **The link + a contact card.** Link their site inline (`sohumai.com`), and repeat name / email /
   phone / web in a footer block so the colleague can act without coming back to ask.
4. **Why now / where to use them.** Name the live projects explicitly and what the outcome should be
   for us — "as we do our content build-out: the websites for Amber, the website for Oasis, and next
   Northstar as well as Vari … highly interactive and utilitarian for our customers and visitors".
   Put the *ask on the colleague* in a highlighted box: get a demonstration, go through the website,
   **give me your feedback**.
5. **A line to the external person** — "marking you here so you and <colleague> can take it forward
   directly", so the thread does not bounce back through NDR.

Sign-off block: `Regards,` / **Nishant Ranka** / `DRA Realty Private Limited · +91 98800 55634`.

Multiple asks across several paragraphs = **HTML with CSS**, not plain text (NDR: plain-text markdown
reads as one continuous running text — BAD). Start from `templates/intro-connection-email.html`.
Always ship a `text/plain` alternative, and make every key point appear in **both** parts.

## 4. Voice-dictated proper nouns — flag, do not "fix"

NDR dictates project and people names by voice. Here "Vari" could not be corroborated against the
contacts snapshot (which holds Ranka Amber / Ranka Oasis / Ranka Northstar / Ranka Iris / Stelo /
Aquagreens). Rules:

- Keep the name **as dictated** in the draft. Do not silently substitute a plausible project.
- Do not block the draft on the clarification either — deliver it and put the uncertainty in the
  **delivery message** as a single line: *"You said 'Vari' — I left it as dictated; tell me if it's a
  different spelling/project name and I'll fix the draft."*
- Same rule for a person's name spelling: state it on the first line of the reply (Sohum vs Soham),
  because it is what NDR types next.

## 5. Build and verify (no attachment)

```python
import sys, base64
sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service
from email.message import EmailMessage

msg = EmailMessage()
msg["To"] = TO; msg["Cc"] = CC; msg["Subject"] = SUBJECT   # ASCII subject
msg.set_content(plain)                    # text/plain
msg.add_alternative(html, subtype="html") # multipart/alternative
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
d = build_service("gmail", "v1", service_name="google-draas") \
      .users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

Then read back through the **draft** endpoint with the policy set (a legacy `Message` has no
`get_body()`), and assert:

```python
from email import message_from_bytes
import email
p = message_from_bytes(base64.urlsafe_b64decode(d["message"]["raw"]), policy=email.policy.default)
assert p.get_body(("plain",)) and p.get_body(("html",))
```

- `LABELS == ['DRAFT']` (never `SENT`)
- `To` / `Cc` / `Subject` byte-exact as intended
- keyword check on **both** parts: each project name, the person's name, the introducer, the website,
  the ask ("demonstration", "feedback"), and the contact-card email/phone

Run `gws_gmail_draft_list` first — if a draft on the same topic already exists, delete it so Drafts
holds exactly one.

## 6. Delivery message to NDR

Bullets, no recap of the email body:

- **To** / **Cc** / **Subject** exactly as drafted
- what the body covers, one line per block (who·origin, the two demos, link+contact card, why-now
  projects, the ask)
- the verification line (DRAFT only, headers exact, both parts present)
- flags (the dictated-name question)
- drafts link: `https://mail.google.com/mail/u/0/#drafts` — plain link, not a code block
