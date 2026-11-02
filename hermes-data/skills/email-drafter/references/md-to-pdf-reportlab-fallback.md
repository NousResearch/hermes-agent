# Markdown → PDF attachments — reportlab fallback (validated 2026-09-14)

Symptom: `scripts/md_to_pdf_dra.py` dies with `ModuleNotFoundError: No module named 'weasyprint'`
on the Hermes VPS. weasyprint is NOT in the venv (pango/cairo system libs missing too).

Durable fix: use `scripts/md_to_pdf_reportlab.py` (reportlab platypus, pure-python wheels).

Install once:
    uv pip install --python /opt/hermes/.venv/bin/python3 reportlab

Run:
    /opt/hermes/.venv/bin/python3 /data/hermes/skills/email-drafter/scripts/md_to_pdf_reportlab.py input.md output.pdf

Capabilities: ATX headings (#..####), pipe tables (auto-detected via
`| a | b |` followed by `| --- |`), "-/* / 1." lists, horizontal rules, paragraphs.
Styling mirrors md_to_pdf_dra.py: navy #1F3864 headings, gold #C99A2E rules,
cream #F5EFDF zebra rows, A4 + page numbers, Helvetica (no font installs).

Verified: Ranka Udaya interior execution spec (1251-line md, 46 sections, many tables)
→ 39 KB clean PDF, attached to the Moduli RFP email draft.

Before attaching any generated PDF:
1. Magic-check the file (`%PDF-` in first bytes) — a zero-byte or truncated PDF
   passes a filename check but renders blank in Gmail.
2. After `drafts().create`, verify attachment BYTES via
   `messages().attachments().get(userId='me', messageId=..., id=attachmentId)`
   and decode base64url (add padding: `4 - (len(data) % 4)`) — the draft's `size`
   field alone is metadata, not proof of content.