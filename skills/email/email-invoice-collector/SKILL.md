---
name: email-invoice-collector
description: "Automatically collect, download, and organize invoices from any IMAP mailbox — supports attachments, in-body download links, ZIP extraction, PDF amount parsing, and Excel/CSV/Markdown report generation."
version: 2.0.0
author: bob798
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Email, Invoice, IMAP, PDF, Finance, Automation, Accounting]
    homepage: https://github.com/bob798/email-assistant
prerequisites:
  commands: [python3, pip]
---

# Email Invoice Collector

Automatically collect electronic invoices from any IMAP mailbox. Scans emails by date range, identifies invoice-related messages via keywords and attachment types, downloads PDF invoices (from attachments, in-body links, or ZIP archives), extracts amounts from PDFs, and generates structured reports.

## Overview

This skill helps you organize invoice emails that are scattered across your mailbox. It handles the full pipeline:

1. **Connect** to any IMAP mailbox (QQ Mail, Gmail, Outlook, 163, etc.)
2. **Identify** invoice emails by subject/body keywords and attachment types
3. **Download** invoice PDFs from attachments, email body links (JD, Tencent Cloud, fapiao.com), or ZIP archives
4. **Extract** amounts from each PDF using pymupdf (handles embedded CID fonts)
5. **Deduplicate** across emails using SHA256 content hashing
6. **Generate** Excel, CSV, and Markdown reports with per-invoice amounts and totals
7. **Mark** successfully processed emails as read

## Prerequisites

- Python 3.10+
- An IMAP-enabled email account with authorization code or app password
- Dependencies: `pip install python-dotenv chardet pymupdf pdfplumber openpyxl pyyaml`

## Installation

```bash
git clone https://github.com/bob798/email-assistant.git
cd email-assistant
pip install -r requirements.txt
cp config.yaml.example config.yaml
# Edit config.yaml with your email credentials
```

## Key Commands

### CLI Usage

```bash
# Dry run — scan only, no downloads, no state changes
python main.py --dry-run

# Full run with default config
python main.py

# Custom date range and output directory
python main.py --since 2026-01-01 --output-dir ./my-invoices

# Use a specific config file
python main.py --config /path/to/config.yaml
```

### Programmatic Usage (for agents/skills)

```python
from core import run

# Basic invocation
result = run(since="2026-05-01")

# With config override (e.g., different mailbox)
result = run(config_override={
    "account": {
        "imap_server": "imap.gmail.com",
        "email": "user@gmail.com",
        "password": "app-password",
    }
})

# Result structure
# {
#   "status": "ok",
#   "stats": {
#     "total_emails": 47,
#     "invoice_emails": 31,
#     "total_attachments": 42,
#     "marked_seen": 24,
#     "errors": 0
#   },
#   "records": [
#     {"date": "2026-05-06", "from_name": "JD", "amount": 46.50, ...},
#     ...
#   ],
#   "report_paths": {
#     "csv": "invoices/summary.csv",
#     "excel": "invoices/summary.xlsx",
#     "markdown": "invoices/report.md"
#   }
# }
```

## Configuration

All settings are in `config.yaml` (see `config.yaml.example`):

- **account** — IMAP server, email, password (supports `${ENV_VAR}` references)
- **search** — date range, folders
- **filter** — keywords, file extensions, format priority
- **amount** — PDF regex patterns, body fallback patterns
- **output** — directory, report formats (csv/excel/markdown)

Falls back to `.env` file if no `config.yaml` is present (backward compatible).

## Supported Email Providers

| Provider | IMAP Server | Notes |
|----------|------------|-------|
| QQ Mail | imap.qq.com | Requires 16-char authorization code |
| Gmail | imap.gmail.com | Requires app password |
| Outlook/365 | outlook.office365.com | Enable IMAP in settings |
| 163 Mail | imap.163.com | Requires authorization code |
| Any IMAP | Custom server | Configure imap_server and imap_port |

## Examples

**Example 1: Monthly invoice collection**
```bash
python main.py --since 2026-05-01
# → invoices/summary.xlsx with all May invoices and total amount
```

**Example 2: Agent integration**
```python
result = run(since="2026-05-01", dry_run=True)
print(f"Found {result['stats']['invoice_emails']} invoice emails")
for r in result['records']:
    print(f"  {r['from_name']}: {r['amount']}")
```

## Troubleshooting

- **IMAP login failed**: Check authorization code (not login password). QQ/163 require special auth codes.
- **SSL certificate errors**: The tool auto-retries with relaxed SSL for platforms with incomplete certificate chains.
- **PDF amount extraction returns None**: Check `config.yaml` pdf_patterns. The tool uses pymupdf (handles CID fonts) with pdfplumber fallback.
- **Duplicate files**: SHA256 content dedup is automatic. Same PDF in different emails is saved only once.
