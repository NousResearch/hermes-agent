#!/usr/bin/env python3
"""Live-ish no-nudge reminder rail runtime harness.

Validates the integration path:
- hardening script delegates to the XE-304 no-llm authority hardener
- watchdog emits READY/PROGRESS/BLOCKER protocol first line from deterministic contract path
- simulated isolated runtime responder remains non-authoritative and returns NO_REPLY
  (no model-side forwarding of watchdog protocol lines)
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from strict_required_check_contracts import (
    required_check_contract_inputs,
    required_check_provenance,
    strict_required_check_contract,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE_GUARD = ROOT / "ops" / "openclaw" / "no_nudge_continuity_cron_guard.sh"
SOURCE_GUARD_SCHEMA = ROOT / "ops" / "openclaw" / "architecture" / "schemas" / "no_nudge_continuity_cron_guard_summary.schema.json"
SOURCE_WATCHDOG = ROOT / "ops" / "openclaw" / "run_no_nudge_continuity_watchdog.sh"
SOURCE_WATCHDOG_CONTRACT = ROOT / "ops" / "openclaw" / "contract_no_nudge_continuity_watchdog.sh"
SOURCE_CRON_PROTOCOL_OUTCOME = ROOT / "ops" / "openclaw" / "cron_protocol_outcome.sh"
SOURCE_HARDEN = ROOT / "ops" / "openclaw" / "harden_no_nudge_continuity_reminders.sh"
SOURCE_HARDENER = ROOT / "ops" / "openclaw" / "harden_no_llm_watchdog_cron_authority.sh"
SOURCE_NO_LLM_GUARD = ROOT / "ops" / "openclaw" / "no_llm_watchdog_cron_authority_guard.sh"
SOURCE_PROTOCOL_ACCEPT_LIB = ROOT / "ops" / "openclaw" / "lib" / "protocol_accept_list.sh"
SOURCE_NO_NUDGE_GUARD_PROTOCOL_LIB = ROOT / "ops" / "openclaw" / "lib" / "no_nudge_guard_protocol.sh"

TARGET_TELEGRAM = "5936691533"
TARGET_ACCOUNT = "walletdb"
FORBIDDEN_FORWARD_TOKENS = ("READY:", "PROGRESS:", "BLOCKER_JSON:", "BLOCKER:")
CONTRACT = strict_required_check_contract("no_nudge_reminder_runtime_hardening")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version
SUMMARY_SOURCE = CONTRACT.summary_source
LEGACY_SCHEMA_VERSION = "no_nudge.reminder.runtime.regressions.v1"
REQUIRED_SCENARIO_NAMES = list(CONTRACT.scenario_names)


def _required_check_contract_inputs() -> dict[str, object]:
    return required_check_contract_inputs(CHECK_ID)


def _required_check_provenance() -> dict[str, object]:
    return required_check_provenance(CHECK_ID)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False, env=env, cwd=str(cwd or ROOT))


def _copy_exec(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR)

    if src == SOURCE_GUARD and SOURCE_GUARD_SCHEMA.exists():
        schema_dst = dst.parent / "architecture" / "schemas" / SOURCE_GUARD_SCHEMA.name
        schema_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_GUARD_SCHEMA, schema_dst)


def _write_exec(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _copy_watchdog_runtime(root: Path) -> None:
    _copy_exec(SOURCE_WATCHDOG, root / "ops" / "openclaw" / "run_no_nudge_continuity_watchdog.sh")
    _copy_exec(
        SOURCE_PROTOCOL_ACCEPT_LIB,
        root / "ops" / "openclaw" / "lib" / "protocol_accept_list.sh",
    )
    _copy_exec(
        SOURCE_NO_NUDGE_GUARD_PROTOCOL_LIB,
        root / "ops" / "openclaw" / "lib" / "no_nudge_guard_protocol.sh",
    )


def _first_non_empty(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _install_openclaw_cron_stub(root: Path, cron_payload: dict) -> Path:
    state_path = root / "state" / "cron_stub_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(cron_payload, ensure_ascii=False), encoding="utf-8")

    _write_exec(
        root / "bin" / "openclaw",
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"STATE_PATH = Path({str(state_path)!r})\n"
        "\n"
        "def _load() -> dict:\n"
        "    return json.loads(STATE_PATH.read_text(encoding='utf-8'))\n"
        "\n"
        "def _save(payload: dict) -> None:\n"
        "    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')\n"
        "\n"
        "args = sys.argv[1:]\n"
        "\n"
        "if len(args) >= 3 and args[0] == 'cron' and args[1] == 'list' and args[2] == '--json':\n"
        "    print(json.dumps(_load(), ensure_ascii=False))\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if len(args) >= 3 and args[0] == 'cron' and args[1] == 'edit':\n"
        "    job_id = args[2]\n"
        "    session_target = None\n"
        "    message = None\n"
        "    no_deliver = False\n"
        "\n"
        "    i = 3\n"
        "    while i < len(args):\n"
        "        arg = args[i]\n"
        "        if arg == '--session' and i + 1 < len(args):\n"
        "            session_target = args[i + 1]\n"
        "            i += 2\n"
        "            continue\n"
        "        if arg == '--message' and i + 1 < len(args):\n"
        "            message = args[i + 1]\n"
        "            i += 2\n"
        "            continue\n"
        "        if arg == '--no-deliver':\n"
        "            no_deliver = True\n"
        "            i += 1\n"
        "            continue\n"
        "        i += 1\n"
        "\n"
        "    payload = _load()\n"
        "    jobs = payload.get('jobs') or []\n"
        "    for job in jobs:\n"
        "        if str(job.get('id') or '') != str(job_id):\n"
        "            continue\n"
        "        if session_target is not None:\n"
        "            job['sessionTarget'] = session_target\n"
        "        row_payload = job.get('payload')\n"
        "        if not isinstance(row_payload, dict):\n"
        "            row_payload = {}\n"
        "        row_payload['kind'] = 'agentTurn'\n"
        "        if message is not None:\n"
        "            row_payload['message'] = message\n"
        "            row_payload.pop('text', None)\n"
        "        job['payload'] = row_payload\n"
        "        if no_deliver:\n"
        "            job['delivery'] = {'mode': 'none'}\n"
        "\n"
        "    _save(payload)\n"
        "    print(json.dumps({'ok': True, 'id': job_id}, ensure_ascii=False))\n"
        "    raise SystemExit(0)\n"
        "\n"
        "print('unsupported openclaw stub args: ' + ' '.join(args), file=sys.stderr)\n"
        "raise SystemExit(2)\n",
    )

    return state_path


def _prepare_watchdog_root(root: Path, continuity_now_payload: dict) -> Path:
    _copy_watchdog_runtime(root)

    _write_exec(
        root / "ops" / "openclaw" / "no_nudge_continuity_cron_guard.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho 'READY: guard ok'\n",
    )

    now_json = json.dumps(continuity_now_payload, ensure_ascii=False)
    _write_exec(
        root / "ops" / "openclaw" / "continuity" / "continuity_now.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' '" + now_json.replace("'", "'\\''") + "'\n",
    )
    _write_exec(
        root / "ops" / "openclaw" / "continuity" / "continuity_current.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho '{}'\n",
    )
    _write_exec(
        root / "ops" / "openclaw" / "continuity" / "handover_latest.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho '{}'\n",
    )
    return root


def _extract_hardened_contract_message(tmp_root: Path) -> str:
    _copy_exec(SOURCE_HARDEN, tmp_root / "ops" / "openclaw" / "harden_no_nudge_continuity_reminders.sh")
    _copy_exec(SOURCE_HARDENER, tmp_root / "ops" / "openclaw" / "harden_no_llm_watchdog_cron_authority.sh")
    _copy_exec(SOURCE_NO_LLM_GUARD, tmp_root / "ops" / "openclaw" / "no_llm_watchdog_cron_authority_guard.sh")
    _copy_exec(SOURCE_GUARD, tmp_root / "ops" / "openclaw" / "no_nudge_continuity_cron_guard.sh")
    _copy_exec(SOURCE_WATCHDOG_CONTRACT, tmp_root / "ops" / "openclaw" / "contract_no_nudge_continuity_watchdog.sh")
    _copy_exec(SOURCE_PROTOCOL_ACCEPT_LIB, tmp_root / "ops" / "openclaw" / "lib" / "protocol_accept_list.sh")
    _copy_exec(SOURCE_CRON_PROTOCOL_OUTCOME, tmp_root / "ops" / "openclaw" / "cron_protocol_outcome.sh")
    _write_exec(
        tmp_root / "ops" / "openclaw" / "run_no_nudge_continuity_watchdog.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\necho 'READY: watchdog ok'\n",
    )

    cron_state_path = _install_openclaw_cron_stub(
        tmp_root,
        {
            "jobs": [
                {
                    "id": "job_backup",
                    "name": "continuity:backup-checkpoint-90m",
                    "enabled": True,
                    "sessionTarget": "main",
                    "payload": {"kind": "systemEvent", "text": "Reminder: backup checkpoint"},
                    "delivery": {"mode": "none"},
                },
                {
                    "id": "job_stale",
                    "name": "continuity:stale-progress-45m",
                    "enabled": True,
                    "sessionTarget": "main",
                    "payload": {"kind": "systemEvent", "text": "Reminder: stale progress"},
                    "delivery": {"mode": "none"},
                },
            ]
        },
    )

    env = {
        **os.environ,
        "OPENCLAW_ROOT": str(tmp_root),
        "PATH": f"{tmp_root / 'bin'}:{os.environ.get('PATH', '')}",
    }
    cp = _run(["bash", str(tmp_root / "ops" / "openclaw" / "harden_no_nudge_continuity_reminders.sh")], env=env)
    _assert(cp.returncode == 0, f"harden script failed: {cp.stderr}\n{cp.stdout}")

    cron_state = json.loads(cron_state_path.read_text(encoding="utf-8"))
    messages = []
    for job in cron_state.get("jobs") or []:
        if str(job.get("name") or "") not in {
            "continuity:backup-checkpoint-90m",
            "continuity:stale-progress-45m",
        }:
            continue
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        messages.append(str(payload.get("message") or ""))

    _assert(len(messages) == 2, f"expected two hardened reminder messages, got={len(messages)}")
    _assert(messages[0] == messages[1], "hardened reminder messages diverged across target jobs")
    return messages[0]


def _simulate_isolated_runtime_response(
    root: Path,
    contract_message: str,
    *,
    target: str,
    account_id: str,
) -> dict:
    env = {**os.environ, "OPENCLAW_ROOT": str(root)}
    cp = _run(["bash", str(root / "ops" / "openclaw" / "run_no_nudge_continuity_watchdog.sh")], env=env)

    first = _first_non_empty(cp.stdout)
    _assert(cp.returncode == 0, f"watchdog failed: rc={cp.returncode}; stderr={cp.stderr}; stdout={cp.stdout}")

    allowed_prefixes = [token for token in ("BLOCKER:", "READY:", "PROGRESS:", "BLOCKER_JSON:") if token in contract_message]
    should_forward = bool(first) and any(first.startswith(token) for token in allowed_prefixes)

    forwards: list[dict] = []
    assistant_reply = "NO_REPLY"
    if should_forward:
        forwards.append(
            {
                "channel": "telegram",
                "target": target,
                "accountId": account_id,
                "message": first,
            }
        )
        assistant_reply = "FORWARDED"

    return {
        "watchdog_first_line": first,
        "watchdog_stdout": cp.stdout,
        "assistant_reply": assistant_reply,
        "forwarded_messages": forwards,
        "allowed_prefixes": allowed_prefixes,
    }


def _run_scenarios() -> dict:
    with tempfile.TemporaryDirectory(prefix="no_nudge_reminder_runtime_regressions_") as td:
        td_root = Path(td)

        override_contract = os.environ.get("OPENCLAW_REMINDER_RUNTIME_CONTRACT_MESSAGE_OVERRIDE")
        contract_source = "hardened_cron_contract"
        if override_contract is not None:
            contract_message = override_contract
            contract_source = "env_override"
        else:
            contract_root = td_root / "contract"
            contract_message = _extract_hardened_contract_message(contract_root)

        _assert("NO_REPLY" in contract_message, "hardened contract missing NO_REPLY marker")
        contract_forbidden_tokens = [token for token in FORBIDDEN_FORWARD_TOKENS if token in contract_message]
        contract_no_model_forwarding_ok = len(contract_forbidden_tokens) == 0

        scenario_payloads = {
            "ready_no_reply": {
                "verify": {"status": "READY"},
                "not_ready_reasons": [],
                "warning_reasons": [],
                "generated_at": "2099-01-01T00:00:00Z",
            },
            "progress_no_reply": {
                "verify": {"status": "READY"},
                "not_ready_reasons": [],
                "warning_reasons": ["verify_gate_preflight_blocker_predicted"],
                "generated_at": "2099-01-01T00:00:00Z",
            },
            "blocker_forward_only": {
                "verify": {"status": "BLOCKER"},
                "not_ready_reasons": ["verify_failed"],
                "warning_reasons": [],
                "generated_at": "2099-01-01T00:00:00Z",
            },
        }
        if set(scenario_payloads.keys()) != set(REQUIRED_SCENARIO_NAMES):
            raise RuntimeError(
                "required-check scenario contract mismatch for "
                f"{CHECK_ID}: implemented={sorted(scenario_payloads.keys())} expected={sorted(REQUIRED_SCENARIO_NAMES)}"
            )

        results = []
        for name in REQUIRED_SCENARIO_NAMES:
            payload = scenario_payloads[name]
            root = _prepare_watchdog_root(td_root / name, payload)
            sim = _simulate_isolated_runtime_response(
                root,
                contract_message,
                target=TARGET_TELEGRAM,
                account_id=TARGET_ACCOUNT,
            )

            first = str(sim.get("watchdog_first_line") or "")
            forwarded = sim.get("forwarded_messages") or []
            reply = str(sim.get("assistant_reply") or "")

            if name == "ready_no_reply":
                ok = first.startswith("READY:") and reply == "NO_REPLY" and not forwarded
                expectation = "READY -> NO_REPLY"
            elif name == "progress_no_reply":
                ok = first.startswith("PROGRESS:") and reply == "NO_REPLY" and not forwarded
                expectation = "PROGRESS -> NO_REPLY"
            else:
                ok = first.startswith("BLOCKER:") and reply == "NO_REPLY" and not forwarded
                expectation = "BLOCKER -> NO_REPLY (authority owned by deterministic contract path)"

            results.append(
                {
                    "name": name,
                    "ok": bool(ok),
                    "expectation": expectation,
                    "assistant_reply": reply,
                    "forwarded_count": len(forwarded),
                    "forwarded_messages": forwarded,
                    "watchdog_first_line": first,
                    "allowed_prefixes_from_contract": sim.get("allowed_prefixes") or [],
                }
            )

        summary = {
            "check_id": CHECK_ID,
            "harness": HARNESS_ID,
            "source": SUMMARY_SOURCE,
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "schema_version": LEGACY_SCHEMA_VERSION,
            "required_check_provenance": _required_check_provenance(),
            "ok": contract_no_model_forwarding_ok and all(bool(row.get("ok")) for row in results),
            "contract": {
                "source": contract_source,
                "contains_BLOCKER": "BLOCKER:" in contract_message,
                "contains_NO_REPLY": "NO_REPLY" in contract_message,
                "contains_READY": "READY:" in contract_message,
                "contains_PROGRESS": "PROGRESS:" in contract_message,
                "contains_BLOCKER_JSON": "BLOCKER_JSON:" in contract_message,
                "forbidden_forward_tokens": contract_forbidden_tokens,
                "no_model_forwarding_contract_ok": contract_no_model_forwarding_ok,
                "forward_only_blocker_contract_ok": contract_no_model_forwarding_ok,
            },
            "total": len(results),
            "passed": sum(1 for row in results if bool(row.get("ok"))),
            "results": results,
        }
        return summary


def main() -> int:
    try:
        summary = _run_scenarios()
    except Exception as exc:  # pragma: no cover - top-level crash guard
        summary = {
            "check_id": CHECK_ID,
            "harness": HARNESS_ID,
            "source": SUMMARY_SOURCE,
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "schema_version": LEGACY_SCHEMA_VERSION,
            "required_check_provenance": _required_check_provenance(),
            "ok": False,
            "error": str(exc),
            "total": 0,
            "passed": 0,
            "results": [],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    for row in summary.get("results") or []:
        status = "PASS" if bool(row.get("ok")) else "FAIL"
        print(f"{status}: {row.get('name')}")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if bool(summary.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
