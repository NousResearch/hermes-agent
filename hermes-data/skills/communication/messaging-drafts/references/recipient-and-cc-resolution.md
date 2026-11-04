# Resolving recipients, and verifying the Cc on a draft

Applies whenever NDR says *"send an email to X, marking A, B, C and D"* — the hard parts are
finding X's actual address from a dictated description, and proving the Cc list landed.

## 1. A dictated person → an email address

NDR describes people by **role and label**, never by address: *"Vinay's contact is stored in my
phone under something like 'Vinay, Licensing, TN RERA Licensing'"*.

The join key is the **saved display name**, which is identical in the phone/WhatsApp contact and
in the Gmail headers:

- phone/WhatsApp contact → `Vinay Prasad TN Liasioning (+91 97400 42068)`
- Gmail headers → `Vinay Prasad TN Liasioning <manconstruction09@gmail.com>`

So: `wa_list_chats(query=...)` or `contact_resolver` gives the saved label → then
`gws_gmail_search(query="<distinctive surname> <project>")` returns From/To headers carrying the
same display name plus the address → take the address from the **most recent thread on that
project**.

**Never resolve by first name alone.** "Vinay" is at least two different men in this mailbox —
Vinay Prasad (TN licensing, Man Construction) and CA Vinay Thyagaraj (RERA Consultants LLP).
The project context plus the saved label is what disambiguates.

## 2. Internal DRA staff frequently have NO email in the contacts sheet

`contact_resolver` is the wrong tool for colleagues — it is a phone-book index:

- `query="Kantesh"` → correct person, **empty `emails` array**.
- `query="Gauri"` → three unrelated candidates, no DRA staffer at all (the person is
  **Gowri Singh**; NDR dictates "Gauri").

Internal addresses come out of Gmail headers instead — search the bare surname or the project
name and read the thread participants. Known ones (verify against a live thread before use):

| Person | Address |
|---|---|
| Anbu / Anbarasan M | pm2.blr@draas.com |
| Sinchana Gowda | sgowda@draas.com |
| Kantesh B G | kanteshbg@draas.com |
| Gowri Singh (dictated "Gauri") | gsingh@draas.com |
| Bharat Hawaldar | sales1.blr@draas.com |
| Sarthak Sharma | admin3.blr@draas.com |

A zero-email `contact_resolver` hit on a colleague is **normal**, not a reason to leave them off
the Cc.

## 3. One person, two live addresses → cc both

Bhuvanesh S Krishnan is addressed on Ranka Oasis threads as **both**
`mail.bskrishnan@gmail.com` and `bk@findingform.design`. When both appear on recent threads, cc
both rather than guessing which he is reading.

## 4. "mark X in the email as well"

Ambiguous phrasing (it can mean *add X to the recipients* or *cc X*). Safe default: the person
NDR said **send to** goes in `To`; everyone he said to **mark** goes in `Cc`. That satisfies both
readings. State the split explicitly in the completion note so he can correct it in one word.

## 5. VERIFY THE Cc — `gws_gmail_get` does not show it

`gws_gmail_get` returns To / From / Subject / Date and **silently omits Cc**, so a draft can look
perfect while the Cc list is empty or truncated. Read the headers directly with the sanctioned
client (`service_name` always explicit):

```python
import sys; sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")
d = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="metadata").execute()
h = {x["name"]: x["value"] for x in d["message"]["payload"]["headers"]}
print("To =>", h.get("To")); print("Cc =>", h.get("Cc"))
```

Print it and check **name by name** against what NDR dictated before reporting the draft ready.
From a shell use `env -u PYTHONPATH /opt/hermes/.venv/bin/python3`.

## 6. A "X must confirm this" email is a request, not a record

When NDR says *"I want this absolutely validated by X"*, *"X needs to confirm this"*, the email
is asking for written confirmation — it is **not** a minute of what X already agreed. Write it as:

1. open with *"recording my understanding of what you have told me"*;
2. number each point so it can be confirmed or rejected individually;
3. ask for confirmation **point by point**, not a blanket yes;
4. close with *"if any part of the above is not correct, say so specifically against that point
   — I would far rather we correct it now than have it surface later"*.

Putting words in a counterparty's mouth in an email copied to the whole team is the failure mode
this framing exists to prevent. The same rule applies to the parallel WhatsApp message: describe
it as *"what we understand"* + *"he must confirm in writing"*, never as settled.
