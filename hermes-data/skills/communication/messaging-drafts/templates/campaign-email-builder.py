"""Campaign email builder — sheet-roster BCC blast, inline CID images, PDF attachment.

TEMPLATE. Copy and modify per campaign. Verified 2026-10-03 (Ranka Amber -> 36
Whitefield channel partners, 19 addresses in BCC, 3 inline renders, 12-page PDF).

Companion doc: messaging-drafts/references/html-body-email-drafting.md

Run it from the /opt/hermes venv with the SESSION USER'S OWN slug, e.g.
    cd /opt/hermes && HERMES_SESSION_USER_ID=psingh-8502281203 \
        /opt/hermes/.venv/bin/python3 campaign_email.py

Hard rules baked in:
  * DRAFT ONLY. This script has no send call and must never gain one.
  * Recipients go in BCC; To stays empty for a blast.
  * Full-width HTML: no max-width anywhere, width:100% on the outer wrapper.
  * Images are downscaled JPEGs, not raw multi-MB PNGs.
"""

import os
import re
import sys
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email import policy, message_from_bytes

from PIL import Image

sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service  # noqa: E402

SERVICE = "google-draas"          # resolve via gws_resolve_account, never guess
OUT = "/opt/data/campaign_out"    # stable path so you can re-verify later
os.makedirs(OUT, exist_ok=True)

SHEET_ID = "PUT_SPREADSHEET_ID_HERE"
SUBJECT = "PUT SUBJECT HERE"

# --------------------------------------------------------------------------
# 1. harvest addresses — regex across EVERY cell, do not trust a column index
# --------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def harvest(sheet_id, rng="A1:Z200"):
    svc = build_service("sheets", "v4", service_name=SERVICE)
    rows = (
        svc.spreadsheets()
        .values()
        .get(spreadsheetId=sheet_id, range=rng)
        .execute()
        .get("values", [])
    )
    found, seen = [], set()
    for idx, row in enumerate(rows[1:], start=2):
        for cell in row:
            for addr in EMAIL_RE.findall(cell or ""):
                a = addr.lower()
                if a not in seen:
                    seen.add(a)
                    label = row[1] if len(row) > 1 else ""
                    firm = row[2] if len(row) > 2 else ""
                    found.append((a, idx, label, firm))
    return rows, found


# --------------------------------------------------------------------------
# 2. downscale renders to inline friendly JPEGs
# --------------------------------------------------------------------------
def prep(src, dst, width, quality=80):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    if w > width:
        im = im.resize((width, int(h * width / w)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=quality, optimize=True)
    return dst


# --------------------------------------------------------------------------
# 3. body — full-width shell, no max-width, cid: refs for the images
# --------------------------------------------------------------------------
def build_html(hero_cid, strip, glance_rows, highlights, attach_note, cta, signature):
    def tr(k, v):
        return (
            '<tr><td style="width:34%;padding:9px 10px;background:#f7f7f5;'
            'border-bottom:1px solid #e6e6e2;"><b>%s</b></td>'
            '<td style="padding:9px 10px;border-bottom:1px solid #e6e6e2;">%s</td></tr>'
            % (k, v)
        )

    glance = "".join(tr(k, v) for k, v in glance_rows)
    hi = "".join(
        '<p style="margin:0 0 8px;">%s</p>' % p for p in highlights
    )
    strip_cells = "".join(
        '<td style="width:%d%%;padding:%s;vertical-align:top;">'
        '<img src="cid:%s" alt="" width="100%%" '
        'style="display:block;width:100%%;height:auto;border:0;"></td>'
        % (100 // len(strip), "0 8px 0 0" if i == 0 else "0 0 0 8px", cid)
        for i, cid in enumerate(strip)
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f4f4f2;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
       style="width:100%;background:#f4f4f2;">
<tr><td align="left" style="padding:0;">
  <img src="cid:{hero_cid}" alt="" width="100%"
       style="display:block;width:100%;height:auto;border:0;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
         style="width:100%;background:#ffffff;">
  <tr><td style="padding:16px 28px 8px;">
    <div style="font-family:Arial,Helvetica,sans-serif;font-size:12px;letter-spacing:2px;
                color:#0f3d2e;border-bottom:2px solid #c9a227;padding-bottom:6px;">
      <b>PROJECT AT A GLANCE</b></div></td></tr>
  <tr><td style="padding:8px 28px 22px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
           style="width:100%;border-collapse:collapse;font-family:Arial,Helvetica,sans-serif;
                  font-size:14px;color:#222222;">{glance}</table></td></tr>
  <tr><td style="padding:8px 28px 20px;font-family:Arial,Helvetica,sans-serif;
                 font-size:14px;line-height:1.7;color:#222222;">{hi}</td></tr>
  <tr><td style="padding:14px 28px 4px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
           style="width:100%;border-collapse:collapse;"><tr>{strip_cells}</tr></table>
  </td></tr>
  <tr><td style="padding:0 28px 22px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
           style="width:100%;background:#f2f7f4;border-left:4px solid #0f3d2e;">
    <tr><td style="padding:16px 18px;font-family:Arial,Helvetica,sans-serif;font-size:14px;
                   line-height:1.6;color:#222222;">{attach_note}</td></tr></table>
  </td></tr>
  <tr><td style="padding:22px 28px;font-family:Arial,Helvetica,sans-serif;font-size:15px;
                 line-height:1.7;color:#ffffff;background:#0f3d2e;">{cta}</td></tr>
  <tr><td style="padding:22px 28px;font-family:Arial,Helvetica,sans-serif;font-size:14px;
                 line-height:1.7;color:#222222;">{signature}</td></tr>
  </table>
</td></tr></table>
</body></html>"""


# --------------------------------------------------------------------------
# 4. assemble + create draft
# --------------------------------------------------------------------------
def main():
    rows, addresses = harvest(SHEET_ID)
    print("addresses found:", len(addresses))
    for a, r, label, firm in addresses:
        print("  row", r, "|", label, "|", firm, "|", a)

    hero = prep(f"{OUT}/hero.png", f"{OUT}/hero.jpg", 1200, 84)
    strip_jpgs = [
        prep(f"{OUT}/strip1.png", f"{OUT}/strip1.jpg", 620, 78),
        prep(f"{OUT}/strip2.png", f"{OUT}/strip2.jpg", 620, 78),
    ]

    html = build_html(
        hero_cid="hero",
        strip=["strip1", "strip2"],
        glance_rows=[("Project", "REPLACE"), ("RERA registration", "REPLACE")],
        highlights=["<b>Point</b> &mdash; replace."],
        attach_note="<b>Attached</b> &mdash; replace with name, version, pages, contents.",
        cta="<b>Next step</b><br>Replace.",
        signature="<b>Sender Name</b><br>Role<br>Phone &middot; email",
    )
    assert html.lower().count("max-width") == 0
    assert html.count("width:100%") >= 1

    msg = MIMEMultipart("mixed")
    msg["Subject"] = SUBJECT
    msg["Bcc"] = ", ".join(a for a, *_ in addresses)  # blast: BCC, empty To

    alt = MIMEMultipart("related")
    alt.attach(MIMEText("PLAIN-TEXT FALLBACK — repeat the key facts here.", "plain"))
    alt.attach(MIMEText(html, "html"))
    for cid, path in [("hero", hero)] + [
        (f"strip{i + 1}", p) for i, p in enumerate(strip_jpgs)
    ]:
        with open(path, "rb") as fh:
            img = MIMEImage(fh.read(), _subtype="jpeg")
        img.add_header("Content-ID", "<%s>" % cid)
        img.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
        alt.attach(img)
    msg.attach(alt)

    with open(f"{OUT}/guide.pdf", "rb") as fh:
        pdf = MIMEApplication(fh.read(), _subtype="pdf")
    pdf.add_header("Content-Disposition", "attachment", filename="REPLACE.pdf")
    msg.attach(pdf)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    print("raw MIME KB:", len(raw) // 1024, "(Gmail ceiling ~25 MB)")

    gmail = build_service("gmail", "v1", service_name=SERVICE)
    print("WHOAMI:", gmail.users().getProfile(userId="me").execute().get("emailAddress"))
    d = (
        gmail.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw}})
        .execute()
    )
    print("DRAFT_ID:", d["id"])
    open("/tmp/campaign_draft_id.txt", "w").write(d["id"] + "\n" + d["message"]["id"])

    # verify off the STORED draft, never the local file
    back = gmail.users().drafts().get(userId="me", id=d["id"], format="raw").execute()
    m = message_from_bytes(base64.urlsafe_b64decode(back["message"]["raw"]),
                           policy=policy.default)
    parts = [(p.get_content_type(), p.get_filename(), p.get("Content-ID"))
             for p in m.walk()]
    print("parts:", parts)
    print("BCC:", (m.get("Bcc") or "")[:120])
    for p in m.walk():
        if p.get_content_type() == "text/html":
            h = p.get_payload(decode=True).decode("utf-8", "ignore")
            print("stored html: max-width", h.lower().count("max-width"),
                  "| width:100%", h.count("width:100%"))


if __name__ == "__main__":
    main()