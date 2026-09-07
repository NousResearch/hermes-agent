"""Local-I/O A/B for #104538; no inference, external APIs, or deliveries.

Run with the project Python and pass a checkout to probe (defaults to this tree).
Exit 1 on the unfixed checkout: the persisted traceback and listed reason are absent.
The injected provider-setup seam performs a real SDK connection to a non-listening
loopback socket; this does not diagnose the original macOS/Gmail incident.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch


def main():
    checkout = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    for key in list(os.environ):
        if key.startswith(("HERMES_", "OPENAI_", "OPENROUTER_", "ANTHROPIC_")) or any(
            part in key for part in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            os.environ.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="cron-diagnostics-") as home:
        os.environ.update(HOME=home, HERMES_HOME=home)
        sys.path.insert(0, str(checkout))
        from cron import jobs, scheduler
        from hermes_cli.cli_commands_mixin import CLICommandsMixin
        from openai import OpenAI
        from tools.cronjob_job_args import _format_job

        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            port = held.getsockname()[1]

            def connection_failure(*args):
                with OpenAI(api_key="local-probe", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0, timeout=1) as client:
                    try:
                        client.chat.completions.create(model="local-probe", messages=[{"role": "user", "content": "probe"}])
                    except Exception as exc:
                        raise RuntimeError("Connection error.") from exc

            job = jobs.create_job(prompt="Check status", schedule="every 1h", model="local-probe", deliver="local")
            with patch.object(scheduler, "_resolve_cron_agent_setup", connection_failure):
                success, output, response, error = scheduler.run_job(job)
            path = jobs.save_job_output(job["id"], output)
            saved = path.read_text(encoding="utf-8")
            jobs.mark_job_run(job["id"], success, error=error)
            formatted = _format_job(jobs.get_job(job["id"]))
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                CLICommandsMixin._cron_list(object(), "list", {"all": False})
            checks = {
                "persisted_traceback": "Traceback (most recent call last)" in saved,
                "persisted_connection_cause": "Connection refused" in saved,
                "listed_run_reason": formatted.get("last_error") == error,
                "slash_list_reason": f"error: {error}" in console.getvalue(),
                "concise_return": not success and not response and error == "RuntimeError: Connection error.",
                "separate_failure_fields": formatted["last_delivery_error"] is None and formatted["last_fire_error"] is None,
                "private_output": os.name == "nt" or path.stat().st_mode & 0o077 == 0,
            }
            script = Path(home) / "scripts" / "healthy.py"
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("print('healthy local control')\n", encoding="utf-8")
            healthy = jobs.create_job(prompt="Local control", schedule="every 1h", script=str(script), no_agent=True, deliver="local")
            ok, _, final, healthy_error = scheduler.run_job(healthy)
            checks["healthy_run"] = ok and "healthy local control" in final and healthy_error is None
            secret_error = "RuntimeError: https://user:password@localhost/api?token=secret-token"
            jobs.mark_job_run(job["id"], False, error=secret_error)
            secret_listing = _format_job(jobs.get_job(job["id"]))
            visible = secret_listing.get("last_error") or ""
            checks["listed_credentials_redacted"] = (
                "localhost" in visible and "password" not in visible and "secret-token" not in visible
            )
            jobs.mark_job_run(job["id"], True)
            checks["success_clears_error"] = jobs.get_job(job["id"])["last_error"] is None
            print(json.dumps({"checkout": str(checkout), "checks": checks}, indent=2))
            assert all(checks.values()), checks


if __name__ == "__main__":
    main()
