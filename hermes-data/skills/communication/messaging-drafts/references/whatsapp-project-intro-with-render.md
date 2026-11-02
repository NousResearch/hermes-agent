# WhatsApp Project-Intro Message + Exterior Render Image

Pattern: introduce a DRA project to a broker/agency by WhatsApp, with the project's
exterior render attached as an image. First done for Ranka Amber -> BRICS Properties
(Mr. Joby Chacko), Oct 2026.

## 1. Resolve the contact — tracker sheet, not the general contacts sheet

Outreach prospects (brokers, agencies, consultants) are **not** in the Google Contacts
sheet. `contact_resolver` will return only fuzzy "index key" noise for them.

- In this deployment `contact_resolver` needs `account_email='psingh@draas.com'` passed
  explicitly — the session user has no default `gws_service`, and omitting it errors with
  *"has no gws_service configured in their profile"*. Passing `'draas'` (the label) is also
  rejected: *"not associated with your session user"*.
- Even with the right account it failed to find Joby Chacko (returned J Srinivasan / Jo Enslin).

**Authoritative source for this outreach class:** the Ranka Amber channel-partner /
agents trackers in the Ranka Amber Drive folder. Search every tab for the name:

| Sheet | id |
|---|---|
| Ranka Amber - Whitefield RERA Agents & Channel Partners - 05Oct2026 | `10lRGQvyCcU8gAk1R3YRr6k1n9nJyBCMHkvIJe9NN1Js` |
| Whitefield Pattandur Agrahara RE Agents & Consultants (PS) 30Sep2026 | `1HkrCkLIgjezJQ7GUyMnMnIt-Roz9YBY5Bv7MZGBM_ZA` |

Tabs on the first: `Summary`, `All Agents & Consultants`, `RERA Registered`,
`RERA Not Published`, `Recruitment Priority`, `Method & Sources`. The second:
`Agents & Consultants`. Rows carry name, firm, RERA reg no, locality, track record,
phone(s), and a priority code — e.g. Joby Chacko (Co-founder), BRICS Properties,
`+91 99450 45451 / 98454 35182`, RERA `PRM/KA/RERA/1251/446/AG/231216/004345`, P2.

Fetch with `build_service("sheets","v4", service_name="google-draas")` under
`HERMES_SESSION_USER_ID=psingh-8502281203` and log a HIT line per matching row
(case-insensitive substring over the joined row against name + firm + STT variants).

## 2. Recover the exact render the intro email used

The intro mails embed the render as a JPEG named e.g. `Ranka Amber — Exterior Render`,
not as a Drive link. To send the identical image, pull it from a sent message.

```python
svc = build_service("gmail", "v1", service_name="google-draas")
m = svc.users().messages().get(userId="me", id=<message_id>).execute()
def walk(p):
    out = []
    if p.get("parts"):
        for x in p["parts"]:
            out += walk(x)
    if p.get("body", {}).get("attachmentId"):
        out.append((p.get("filename"), p["mimeType"], p["body"]["attachmentId"]))
    return out
for fn, mt, aid in walk(m["payload"]):
    a = svc.users().messages().attachments().get(
        userId="me", messageId=<message_id>, id=aid).execute()
    open("/opt/data/intro_render.jpg", "wb").write(base64.urlsafe_b64decode(a["data"]))
```

**Pitfall — attachment tokens are short-lived.** The `attachmentId` printed by a
`gws_gmail_get` tool call is a one-time token. Reusing it in a later script raises
HTTP 400 `"Invalid attachment token"`. Always re-fetch the message and walk the payload
**inside the same script**, then download immediately. Never copy an attachmentId
from an earlier turn's output.

Drive-side renders live in the `Renders` folder (`1bPyOjADfxl2E2nQUjnYP8MgBHhVjpCBR`):
`Exterior 1-4.png`, `Lobby 1-2.png`, `Gym 1-2.png`, `Terrace 1-2.png` (1537x1023 PNGs).
`Exterior 1.png` (`1q9xbczx_1gdR8ByWF_HddpVFU2DFLdLv`) is the DRA-branded hero elevation
(stilt parking + 4 floors + rooftop terrace, "DRA HOME OF PRIDE" signage). Download with
`svc.files().get_media(fileId=...)`.

## 3. Deep link is TEXT-ONLY — deliver the image separately

`whatsapp_link` (or any api.whatsapp.com URL) pre-fills text only. It cannot carry an
image. Do not promise an "image-attached" link.

Workflow:
1. `whatsapp_link(phone=..., text=..., platform='telegram')` -> deliver `display_link`.
2. `send_message(target='telegram', message='...MEDIA:/opt/data/intro_render.jpg')` so the
   user has the file on their phone.
3. Tell the user plainly: attach the image to the chat themselves after the text pre-fills.

State honestly that nothing has been sent — the link only opens a pre-filled compose box.

## 4. Message shape (work register, external agency)

Bold caption first line, then salutation, project paragraph, a bulleted fact block pulled
verbatim from the intro email, the document/empanel invitation, then the Prakash Singh
signature block. Mirror the same facts as the email so both channels read identically:
RERA reg no, sanctioned plan no, units/areas, base price, possession target, bank APF.