#!/usr/bin/env bash
# extract_cookies_help.sh — Print step-by-step Google cookie extraction instructions
# for NotebookLM manual authentication. No side effects.
set -euo pipefail

cat <<'EOF'
=== NotebookLM — Manual cookie extraction ===

NotebookLM has no official API. Authentication uses Google browser cookies.
Extract them on a machine with Chrome, then import on the server.

STEP 1 — Open Chrome and go to https://notebooklm.google.com
          Make sure you are logged in to your Google account.

STEP 2 — Press F12 (or Cmd+Option+I on Mac) to open DevTools.

STEP 3 — Click the "Network" tab.

STEP 4 — In the filter box, type: batchexecute

STEP 5 — Click on any notebook to trigger a request.

STEP 6 — Click on a "batchexecute" request in the list.

STEP 7 — In the right panel, scroll to "Request Headers".

STEP 8 — Find the line starting with: cookie:

STEP 9 — Right-click the cookie VALUE and select "Copy value".

STEP 10 — Paste into a text file and save as cookies.txt.
          Format: SID=abc...; HSID=xyz...; SSID=...; APISID=...; SAPISID=...; __Secure-1PSID=...; ...
          Lines starting with # are comments and are ignored.

STEP 11 — Copy cookies.txt to the server and run:
          nlm login --manual --file cookies.txt

STEP 12 — Verify:
          nlm login --check
          nlm doctor

ALTERNATIVE (OpenClaw CDP — if a managed browser is running on the server):
          nlm login --provider openclaw --cdp-url http://127.0.0.1:18800

ALTERNATIVE (desktop auto mode):
          nlm login
          # then copy ~/.notebooklm-mcp-cli/profiles/default/auth.json to the server
EOF
