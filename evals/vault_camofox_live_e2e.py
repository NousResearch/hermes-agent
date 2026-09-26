"""Opt-in, synthetic-only vault fill against a real Camofox server.

Run with the Hermes interpreter, passing --env-file for CAMOFOX_URL/API_KEY.
Uses a disposable HERMES_HOME and unique remote browser identity, never the real
vault/profile. No login or form is submitted. Only booleans/counts leave the test.
--source permits running the same checks against an unpatched checkout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--audit-ssh-host",
        help="Optional SSH target for secret-free remote journal check",
    )
    parser.add_argument("--audit-ssh-key", type=Path)
    parser.add_argument("--audit-known-hosts", type=Path)
    args = parser.parse_args()
    if args.audit_ssh_host and not (args.audit_ssh_key and args.audit_known_hosts):
        parser.error("Remote audit needs an explicit identity and known-hosts file")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sys.path.insert(0, str(args.source.resolve()))
    from dotenv import dotenv_values

    settings = dotenv_values(args.env_file)
    for key in ("CAMOFOX_URL", "CAMOFOX_API_KEY"):
        value = os.environ.get(key) or settings.get(key)
        if not value:
            print(json.dumps({"success": False, "missing_setting": key}))
            return 2
        os.environ[key] = value
    del settings
    identity = "vault-e2e-" + secrets.token_hex(8)
    os.environ["CAMOFOX_USER_ID"] = identity
    os.environ["CAMOFOX_SESSION_KEY"] = identity
    os.environ["CAMOFOX_ADOPT_EXISTING_TAB"] = "false"
    os.environ.pop("BROWSER_CDP_URL", None)
    os.environ.pop("HERMES_PROFILE", None)
    task = identity
    checks: dict[str, bool] = {}
    canary = "synthetic-only-" + secrets.token_urlsafe(32)

    with tempfile.TemporaryDirectory(
        prefix="vault-camofox-e2e-", dir=os.environ.get("TMPDIR")
    ) as home:
        os.environ["HERMES_HOME"] = home
        Path(home, "config.yaml").write_text(
            json.dumps({
                "browser": {
                    "cloud_provider": "camofox",
                    "camofox": {
                        "managed_persistence": False,
                        "adopt_existing_tab": False,
                    },
                },
            }),
            encoding="utf-8",
        )
        from agent.vault_store import get_vault_store
        from tools import browser_camofox as camo
        from tools import browser_vault_tool as vault
        from tools import browser_tool_session
        from tools.browser_tool import browser_console

        checks["isolated_vault"] = get_vault_store()._base == Path(home, "vault")
        assert checks["isolated_vault"], "E2E vault must be disposable"
        checks["camofox_selected"] = camo.is_camofox_mode()
        local_calls: list[str] = []

        def refuse_local(*_args, **_kwargs):
            local_calls.append("local_browser_called")
            return {"success": False, "error": "local browser forbidden by E2E guard"}

        def evaluate(expression: str):
            session = camo._get_session(task)
            return camo._post(
                camo._tab_path(session, "evaluate"),
                {
                    "userId": session["user_id"],
                    "expression": expression,
                },
            ).get("result")

        def assert_check(name: str, ok: bool):
            checks[name] = bool(ok)
            if not ok:
                raise AssertionError(name)

        def inject_form():
            evaluate("""(() => {
              document.body.innerHTML = '<form onsubmit="return false"><input name="user" autocomplete="username"><input name="pass" type="password" autocomplete="current-password"></form>';
              return true;
            })()""")

        def field_matches():
            # Camoufox's Xray sandbox forbids cross-realm TypedArray hashing.
            # Compare the synthetic-only value inside this process; never print it.
            return evaluate("document.querySelector('[name=pass]').value") == canary

        try:
            with patch.object(
                browser_tool_session, "_run_browser_command", refuse_local
            ):
                nav = json.loads(
                    camo.camofox_navigate("https://example.com", task_id=task)
                )
                assert_check("remote_navigation", nav.get("success") is True)
                assert_check(
                    "isolated_remote_identity",
                    camo._get_session(task)["user_id"] == identity,
                )
                inject_form()
                item = get_vault_store().add_item(
                    "login",
                    "Synthetic E2E",
                    {
                        "identifier_type": "username",
                        "identifier": "synthetic-user",
                        "password": canary,
                    },
                    origin="https://example.com",
                )
                raw = vault.browser_vault_fill(item.id, task_id=task)
                out = json.loads(raw)
                assert_check("fill_response_secret_free", canary not in raw)
                assert_check(
                    "login_fill",
                    out.get("success") is True and out.get("filled_fields") == 1,
                )
                assert_check("exact_dummy_value_landed", field_matches() is True)
                assert_check(
                    "identifier_untouched",
                    evaluate("document.querySelector('[name=user]').value === ''")
                    is True,
                )
                readback = browser_console(
                    expression="document.querySelector('[name=pass]').value",
                    task_id=task,
                )
                assert_check(
                    "later_model_readback_redacted",
                    canary not in readback,
                )

                wrong = get_vault_store().add_item(
                    "login",
                    "Wrong origin",
                    {
                        "identifier_type": "username",
                        "identifier": "synthetic-user",
                        "password": canary,
                    },
                    origin="https://example.org",
                )
                inject_form()
                out = json.loads(vault.browser_vault_fill(wrong.id, task_id=task))
                assert_check(
                    "wrong_origin_refused", out.get("error_type") == "origin_mismatch"
                )
                assert_check(
                    "wrong_origin_wrote_nothing",
                    evaluate("document.querySelector('[name=pass]').value === ''")
                    is True,
                )

                # Change origin between classification and the real secret transport.
                original = vault._eval_js_secret

                def navigate_before_fill(task_id, expression, **kwargs):
                    nav = json.loads(
                        camo.camofox_navigate("https://example.org", task_id=task_id)
                    )
                    assert nav.get("success") is True
                    inject_form()
                    return original(task_id, expression, **kwargs)

                inject_form()
                with patch.object(vault, "_eval_js_secret", navigate_before_fill):
                    out = json.loads(vault.browser_vault_fill(item.id, task_id=task))
                assert_check(
                    "navigation_race_refused", out.get("error_type") == "origin_changed"
                )
                assert_check(
                    "navigation_race_wrote_nothing",
                    evaluate("document.querySelector('[name=pass]').value === ''")
                    is True,
                )
                assert_check("no_local_browser_calls", not local_calls)
        except Exception as exc:
            print(
                json.dumps({
                    "success": False,
                    "checks": checks,
                    "failure_type": type(exc).__name__,
                    "http_status": getattr(
                        getattr(exc, "response", None), "status_code", None
                    ),
                    "local_browser_calls": len(local_calls),
                })
            )
            return 1
        finally:
            # Clear DOM before closing, then remove only this test's remote session.
            try:
                evaluate("document.body.replaceChildren(); true")
            except Exception:
                pass
            cleanup = json.loads(camo.camofox_close(task_id=task))
            try:
                remaining = camo._get("/tabs", params={"userId": identity}).get(
                    "tabs", []
                )
                checks["remote_test_tabs_removed"] = (
                    not remaining and "warning" not in cleanup
                )
            except Exception:
                checks["remote_test_tabs_removed"] = False
        if args.audit_ssh_host:
            # Never return remote log text. The synthetic canary travels only via stdin.
            scanner = (
                "import sys,json,subprocess; d=json.load(sys.stdin); "
                "p=subprocess.run(['journalctl','-u','camofox-browser','--since',d['since'],"
                "'--no-pager','--output=cat'],capture_output=True,text=True); "
                "print(json.dumps({'success':p.returncode==0 and 'evaluate' in p.stdout "
                "and 'not seeing' not in p.stderr.lower() and 'permission' not in p.stderr.lower(),"
                "'canary_absent':d['canary'] not in p.stdout and d['canary'] not in p.stderr}))"
            )
            command = [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                f"UserKnownHostsFile={args.audit_known_hosts}",
                "-i",
                str(args.audit_ssh_key),
                args.audit_ssh_host,
                "python3 -c " + shlex.quote(scanner),
            ]
            try:
                audit = subprocess.run(
                    command,
                    input=json.dumps({"since": started, "canary": canary}),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                result = json.loads(audit.stdout)
                checks["remote_journal_canary_absent"] = (
                    audit.returncode == 0
                    and result.get("success") is True
                    and result.get("canary_absent") is True
                )
            except Exception:
                checks["remote_journal_canary_absent"] = False
        print(
            json.dumps(
                {"success": all(checks.values()), "checks": checks}, sort_keys=True
            )
        )
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
