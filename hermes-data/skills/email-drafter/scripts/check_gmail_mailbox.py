#!/usr/bin/env python3
"""Probe: which mailbox does this terminal session's GWS identity resolve to?

In cron/background sessions HERMES_SESSION_USER_ID can be stale and map to a
DIFFERENT user's vault token (e.g. psingh@draas.com instead of ndr@draas.com),
silently making Gmail searches return 0 results. Run this to confirm before any
Gmail op. If the mailbox is wrong, re-run with the owner prefix:

    HERMES_SESSION_USER_ID=<owner-id-from-HERMES_CRON_JOB_OWNER_ID> python3 check_gmail_mailbox.py

Usage: cd /opt/hermes && /opt/hermes/.venv/bin/python3 /opt/hermes/.../check_gmail_mailbox.py [service_name]
"""
import os
import sys

service_name = sys.argv[1] if len(sys.argv) > 1 else "google-draas"

print("ENV USER_ID:", os.environ.get("HERMES_SESSION_USER_ID"))
try:
    from gateway.session_context import get_gws_identity_env
    print("GWS_IDENTITY_ENV:", get_gws_identity_env())
except Exception as e:
    print("session_context err:", e)

from tools.gws_auth import build_service  # noqa: E402

gmail = build_service("gmail", "v1", service_name=service_name)
print("MAILBOX:", gmail.users().getProfile(userId="me").execute()["emailAddress"])