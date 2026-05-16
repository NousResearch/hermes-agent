#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

DOCS = {
    "readme": Path("/usr/local/lib/hermes-agent/README_DOCS.md"),
    "index": Path("/usr/local/lib/hermes-agent/NODE_DOCUMENT_INDEX_2026-05-13.md"),
    "router": Path("/usr/local/lib/hermes-agent/RUNTIME_AUTHORITY_OPERATOR_ENTRY_2026-05-13.md"),
    "register": Path("/usr/local/lib/hermes-agent/RUNTIME_EXCEPTION_REGISTER_AND_AUTHORITY_MAP_2026-05-13.md"),
    "checklist": Path("/usr/local/lib/hermes-agent/RUNTIME_VERIFY_CHECKLIST_2026-05-13.md"),
    "recovery": Path("/usr/local/lib/hermes-agent/RCN_006_RECOVERY_AND_ROLLBACK_CONTRACT_2026-05-13.md"),
    "backlog": Path("/usr/local/lib/hermes-agent/MASTER_BACKLOG_MATRIX_LIMB_2026-05-13.md"),
    "onepage": Path("/usr/local/lib/hermes-agent/GOVERNANCE_ONEPAGE_2026-05-13.md"),
    "closure": Path("/usr/local/lib/hermes-agent/ARTEMIDA_EXCEPTION_CLOSURE_CHECKLIST_2026-05-13.md"),
}

REQUIRED_PHRASES = {
    "readme": [
        "NODE_DOCUMENT_INDEX_2026-05-13.md` as the highest document-routing and precedence authority",
        "RUNTIME_AUTHORITY_OPERATOR_ENTRY_2026-05-13.md` as the first-stop runtime router",
        "ARTEMIDA authority cluster",
    ],
    "index": [
        "last_verified: 2026-05-16",
        "ARTEMIDA authority cluster",
        "exception-closure-checklist — active-exception",
        "closure checklist owning retirement/recreate eligibility",
        "ARTEMIDA_NORMALIZATION_ASSESSMENT_2026-05-13.md` as review-needed historical context only",
    ],
    "router": [
        "last_verified: 2026-05-16",
        "Runtime observation is highest authority for current factual state.",
        "Exception register is highest authority for class assignment and precedence map.",
    ],
    "register": [
        "last_verified: 2026-05-16",
        "- `artemida-osint`",
        "1. Narrow exception/register authority for its explicit scope",
    ],
    "checklist": [
        "last_verified: 2026-05-16",
        "the only named architectural exception is `artemida-osint`.",
        "any separately documented governed add-on workload",
    ],
    "recovery": [
        "last_verified: 2026-05-16",
        "Routine recreate: NOT allowed.",
        "controlled recreate of `artemida-osint` only after explicit prerequisite validation",
    ],
    "backlog": [
        "last_verified: 2026-05-16",
        "one named frozen exception remains: `artemida-osint`",
    ],
    "onepage": [
        "last_verified: 2026-05-16",
        "1. `ARTEMIDA_EXCEPTION_CLOSURE_CHECKLIST_2026-05-13.md`",
        "4. `ARTEMIDA_LEGACY_2080_DISPOSITION_2026-05-13.md`",
    ],
    "closure": [
        "lifecycle: active-exception",
        "preferred_use",
        "successor_policy",
        "last_verified: 2026-05-16",
    ],
}

FORBIDDEN_PHRASES = {
    "index": [
        "ARTEMIDA active-exception authority cluster",
    ],
}

SYSTEMD_UNITS = ["docker", "containerd", "systemd-journald", "rsyslog", "tailscaled"]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": cmd, "code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


result = {"status": "PASS", "docs": {}, "runtime": {}, "failures": []}

for key, path in DOCS.items():
    entry = {"path": str(path), "exists": path.exists(), "missing_required": [], "forbidden_found": []}
    if not path.exists():
        result["status"] = "FAIL"
        result["failures"].append(f"missing doc: {path}")
        result["docs"][key] = entry
        continue
    text = path.read_text()
    for phrase in REQUIRED_PHRASES.get(key, []):
        if phrase not in text:
            entry["missing_required"].append(phrase)
            result["status"] = "FAIL"
            result["failures"].append(f"missing phrase in {path.name}: {phrase}")
    for phrase in FORBIDDEN_PHRASES.get(key, []):
        if phrase in text:
            entry["forbidden_found"].append(phrase)
            result["status"] = "FAIL"
            result["failures"].append(f"forbidden phrase in {path.name}: {phrase}")
    result["docs"][key] = entry

ps = run(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Networks}}"])
result["runtime"]["docker_ps"] = ps
if ps["code"] != 0:
    result["status"] = "FAIL"
    result["failures"].append("docker ps failed")
else:
    lines = [line for line in ps["stdout"].splitlines() if line.strip()]
    result["runtime"]["docker_ps"]["count"] = len(lines)
    if len(lines) != 10:
        result["status"] = "FAIL"
        result["failures"].append(f"expected 10 running containers, got {len(lines)}")
    for expected in ["artemida-osint", "showcase-agui-pod", "commander-osint-pod", "audit-omlite-pod", "venom-1047-pod"]:
        if not any(line.startswith(expected + "\t") for line in lines):
            result["status"] = "FAIL"
            result["failures"].append(f"missing running container: {expected}")
    wrong_network = [line for line in lines if "docker-pods_default" not in line]
    if wrong_network:
        result["status"] = "FAIL"
        result["failures"].append(f"containers outside docker-pods_default: {wrong_network}")

for unit in SYSTEMD_UNITS:
    active = run(["systemctl", "is-active", unit])
    enabled = run(["systemctl", "is-enabled", unit])
    result["runtime"].setdefault("systemd", {})[unit] = {"active": active, "enabled": enabled}
    if active["stdout"] != "active":
        result["status"] = "FAIL"
        result["failures"].append(f"unit not active: {unit}={active['stdout']}")
    if unit == "systemd-journald":
        if enabled["stdout"] not in {"static", "enabled"}:
            result["status"] = "FAIL"
            result["failures"].append(f"journald unexpected enablement state: {enabled['stdout']}")
    else:
        if enabled["stdout"] not in {"enabled", "static"}:
            result["status"] = "FAIL"
            result["failures"].append(f"unit not enabled/static: {unit}={enabled['stdout']}")

print(json.dumps(result, indent=2, ensure_ascii=False))
