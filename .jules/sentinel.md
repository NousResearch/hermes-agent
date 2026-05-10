## 2024-05-18 - Input Length Limits
**Vulnerability:** Missing input length validation on command endpoints could allow Denial of Service (DoS) via excessively long payloads.
**Learning:** `tui_gateway/server.py` relies on `subprocess.run(shell=True)` for intentional remote shell execution. Instead of disabling the shell (which would break functionality), adding length constraints limits resource exhaustion.
**Prevention:** Always enforce a maximum length (e.g., 4096 chars) on inputs that are passed to subprocess execution or external APIs.
