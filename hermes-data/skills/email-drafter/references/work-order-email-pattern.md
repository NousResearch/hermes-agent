# Work-Order Email Pattern (replicate an existing sent HTML email + attach an image)

Class: NDR asks for a "work order email for <person>, exact same HTML formatting as the one we sent <other person> for <project>".

## Steps (validated 2026-10-12: Ruhaan 914 WO → Rivaan 914 bathroom WO)

1. **Find the prior email** — `gws_gmail_search(query="work order <recipient>", ...)` or `"<project> work order"`. The 914 WO was `*914 Embassy Habitat — Work Order: Maker Space & Home Office Design*`, sent from ndr@draas.com to Ruhaan (pebblyshark69@gmail.com).
2. **Pull the exact HTML template** from the sent message, not from memory of the text:
   ```python
   svc = build_service("gmail", "v1", service_name="google-draas")
   msg = svc.users().messages().get(userId="me", id=MSG_ID, format="full").execute()
   for p in msg["payload"].get("parts", []):
       if p.get("mimeType") == "text/html":
           html = base64.urlsafe_b64decode(p["body"]["data"] + "=" * (-len(p["body"]["data"]) % 4)).decode()
           open("/tmp/template.html", "w").write(html)
   ```
   Save it as the skeleton — keep the exact structure: `#1d1d1f` summary box with `#f5a623` uppercase label, `h3` with `border-bottom:2px solid #f5a623`, numbered sections, `<b>` for ₹ figures and key terms.
3. **Adapt content** per dictation. NDR's WO shape: Section 1 Objective & Tools; 2 Functional Requirements (with an APPROVAL GATE step where required); 3 Deliverables D1–D9; 4 Methodology & Budget discipline (cheap models first, premium only once design locked); 5 Quality/Acceptance; 6 Payment (fee + the "normally ₹X but paying ₹Y to encourage you" line for kids' first jobs); 7 Timeline; 8 Coordination (who is in CC and why). Keep the "Reply to this mail to accept the work order" close.
4. **Attach an image** via raw MIME — native/bridge draft tools have no attachment parameter:
   ```python
   from email.mime.image import MIMEImage   # images → MIMEImage, _subtype="jpeg" → image/jpeg
   msg = MIMEMultipart("mixed")
   msg.attach(MIMEText(html, "html", "utf-8"))
   att = MIMEImage(open(path, "rb").read(), _subtype="jpeg")
   att.add_header("Content-Disposition", "attachment", filename=ATT_NAME)
   msg.attach(att)
   svc.users().drafts().create(userId="me", body={"message": {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}})
   ```
   **Pitfall:** `MIMEApplication(bytes, _subtype="jpeg")` yields `application/jpeg` — wrong content type. Use MIMEImage for images; MIMEApplication only for PDFs/octet-stream.
5. **BYTE-VERIFY the attachment**: sha256 of the embedded part must equal sha256 of the local file (presence in parts metadata is not proof).
6. **Delete any stale draft** for the same recipient/topic before creating the new one (`drafts().delete`), so the Drafts folder carries exactly one.
7. **Verify raw MIME** of the created draft: To / Cc / Subject / parts — phantom-success rule (see native-gws-tools-workflow.md).
8. DRAFT ONLY — NDR taps send. Report the draft link.

## Sending-account rule
Personal/family jobs go on the personal vault account (`nishantranka@gmail.com` / google-gmail); company WOs on google-draas. NDR: "Send it for my personal email because it's a personal job."

## Voice-name pitfalls (2026-10-12)
- "Mark Pebley / Pebli" = **Ruhaan** (pebblyshark69@gmail.com) — "Pebby" from pebbly-shark username.
- "Akber" is ambiguous by context: M. Akber Hussain = the Millers Road landowner's GPA holder (lease deeds), vs "Akber/Aamir Khan" = DRA architect on Miller's Road office. Resolve from doc context, not by name alone.
- Kids' emails: Rivaan = rankarivaan@gmail.com (NDR's sent mail), BUT Roshni's calendar invites list rivaanranka@gmail.com — confirm which address to use if it matters; Ruhaan = pebblyshark69@gmail.com.