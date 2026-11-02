#!/usr/bin/env python3
"""Batch WhatsApp links compiled into a .docx with clickable hyperlinks.

Worked pattern (30-Aug-2026): 32 Kelsa SSV leads -> per-lead drafted message
-> one Word document where each tap opens WhatsApp with the number + pre-filled text.

Usage:
  1. Fill LEADS below: (name, phone, honorific, message, context_note)
  2. Run with the Hermes venv:
       /opt/hermes/.venv/bin/python3 whatsapp-links-docx.py
     (install python-docx first if missing:
        uv pip install --python /opt/hermes/.venv/bin/python3 python-docx)
  3. Verify output: script prints hyperlink count + sets 'bad' list.
  4. Deliver via MEDIA:/path/to/file.docx

Message rules (Bharat, DRAAS):
  - Address customer as "Ma'am" (ladies) / "Sir" (gents) — NEVER the first name.
  - Draft each message from the source record's LAST note (conversation state).
  - No raw "&" in message text (whatsapp_link_tool handles encoding; & becomes fullwidth).
"""
import sys, json
sys.path.insert(0, "/opt/hermes")
from tools.whatsapp_link_tool import whatsapp_link_tool

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# name, phone, honorific (Sir/Ma'am), message, context note
LEADS = [
    ("Example Lead", "91XXXXXXXXXX", "Sir", "Hello Sir, ...", "Last note recap."),
]

def add_hyperlink(paragraph, url, text):
    """python-docx has no native hyperlink API — build w:hyperlink with a rels relationship."""
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    color = OxmlElement("w:color"); color.set(qn("w:val"), "0563C1"); rPr.append(color)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rPr.append(u)
    new_run.append(rPr)
    t = OxmlElement("w:t"); t.text = text; new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(10.5)

doc.add_heading("Lead Follow-ups — WhatsApp Messages", level=0)

failures = []
for i, (name, phone, honorific, msg, ctx) in enumerate(LEADS, 1):
    doc.add_heading(f"{i}. {name}", level=1)
    p = doc.add_paragraph()
    p.add_run(f"📞 {phone}    |    Address: {honorific}").bold = True
    pc = doc.add_paragraph()
    rc = pc.add_run("Last note: " + ctx)
    rc.italic = True; rc.font.size = Pt(9); rc.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    pm = doc.add_paragraph(); pm.add_run("Message:").bold = True
    pb = doc.add_paragraph()
    pb.paragraph_format.left_indent = Inches(0.3)
    pb.add_run(msg).font.size = Pt(10)
    pl = doc.add_paragraph()
    pl.add_run("WhatsApp link: ").bold = True
    try:
        res = json.loads(whatsapp_link_tool({"phone": phone, "text": msg, "platform": "telegram"}))
        add_hyperlink(pl, res["url"], f"Open WhatsApp — {name}")
    except Exception as e:
        failures.append((name, str(e)))

out = "/opt/data/WhatsApp_Followups.docx"
doc.save(out)

# Verify: every hyperlink target must be a real api.whatsapp.com/send URL.
from docx.oxml.ns import qn as _qn
hyps = [h.get(_qn('r:id')) for p in doc.paragraphs for h in p._p.findall(_qn('w:hyperlink'))]
bad = [p.part.rels[rid].target_ref for rid in hyps
       if not p.part.rels[rid].target_ref.startswith("https://api.whatsapp.com/send?")]
print(f"Saved: {out}")
print(f"Leads: {len(LEADS)}, links: {len(hyps)}, bad URLs: {len(bad)}, failures: {len(failures)}")
for f in failures:
    print("FAIL:", f)