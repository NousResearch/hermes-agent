#!/usr/bin/env python3
"""Claude Code PreToolUse hook: the mechanical half of NOVA's Decision/Safety layer.

Reads the hook event JSON on stdin and answers with a permission decision:
  deny -> never allowed for an agent (destroy, secret exposure, public ingress...)
  ask  -> allowed only with a human clicking approve (saved-plan apply, --prune, state surgery)
  (no output) -> normal permission flow

Also usable directly for testing:  echo '{"tool_name":"Bash","tool_input":{"command":"terraform destroy"}}' | command_guard.py
"""
import json
import re
import sys

DENY = [
    (r"tf\.sh\s+destroy\b", "destroy is never a troubleshooting or rollback step"),
    (r"\bterraform\b.*\bdestroy\b", "terraform destroy is never a troubleshooting or rollback step"),
    (r"\bterraform\b.*\bapply\b.*-destroy\b", "apply -destroy is equivalent to destroy"),
    (r"\bterraform\b(?!.*\bshow\b).*\bapply\b", "raw terraform apply bypasses plan_guard; use scripts/tf.sh plan then scripts/tf.sh apply"),
    (r"hashicorp/terraform\S*\s+.*\bapply\b", "raw dockerised terraform apply bypasses plan_guard; use scripts/tf.sh apply"),
    (r"\bterraform\b.*-auto-approve", "-auto-approve skips review"),
    (r"\bterraform\b.*\bforce-unlock\b", "force-unlock can corrupt state; human only"),
    (r"\baws\s+ec2\s+terminate-instances\b", "terminating the runtime destroys a customer environment"),
    (r"\baws\s+ec2\s+(delete-volume|detach-volume)\b", "EBS state volume is protected"),
    (r"\baws\s+ec2\s+delete-snapshot\b", "snapshots may be the only backup"),
    (r"\baws\s+kms\s+(schedule-key-deletion|disable-key)\b", "disabling/deleting KMS keys makes data unrecoverable"),
    (r"\baws\s+s3\s+rb\b|\baws\s+s3api\s+delete-bucket\b", "bundle bucket is persistent deployment data"),
    (r"\baws\s+s3\s+rm\b.*--recursive", "recursive S3 delete of deployment data"),
    (r"\baws\s+ecr\s+(delete-repository|batch-delete-image)\b", "deleting images breaks rollback"),
    (r"\baws\s+iam\s+(delete-role|delete-role-policy|detach-role-policy)\b", "IAM changes go through Terraform plan review"),
    (r"\baws\s+iam\s+(put-role-policy|attach-role-policy|create-access-key)\b", "IAM changes go through Terraform; no long-lived keys"),
    (r"\baws\s+logs\s+delete-log-group\b", "log groups are the audit trail"),
    (r"authorize-security-group-ingress\b.*(0\.0\.0\.0/0|::/0)", "public ingress is not allowed for convenience"),
    (r"\baws\s+ec2\s+associate-address\b", "runtime must stay private"),
    (r"(cat|less|more|head|tail|type)\s+\S*\.aws/credentials", "never print credentials"),
    (r"\baws\s+configure\s+get\s+\S*(secret|session_token|access_key)", "never print credentials"),
    (r"\benv\b\s*\|\s*grep\s+-i\s+aws|printenv\s+AWS_SECRET", "never print credentials"),
]
ASK = [
    (r"scripts/tf\.sh\s+apply\b", "applying a reviewed saved plan — confirm plan sha and safety verdict"),
    (r"\bterraform\b.*\bstate\s+(rm|mv|push|replace-provider)\b", "state surgery changes what Terraform manages"),
    (r"\bterraform\b.*\bimport\b", "import changes state ownership"),
    (r"-replace=", "explicit replacement — confirm backup/snapshot and restore test"),
    (r"\bnova\b.*\bapply\b.*--prune\b", "--prune removes profiles not in bundle — confirm the diff"),
    (r"\baws\s+ec2\s+(stop|reboot)-instances\b", "interrupts the customer runtime"),
    (r"\baws\s+ssm\s+send-command\b.*(docker\s+(rm|rmi|system\s+prune|volume\s+rm)|rm\s+-rf|systemctl\s+(stop|restart))", "mutating remote command on runtime"),
    (r"\bdocker\s+push\b", "pushing an image — confirm tests/scan passed"),
]

PROTECT_MARKERS = ("prevent_destroy", "ignore_changes")


def decide_bash(cmd):
    c = " ".join(cmd.split())
    # scripts/tf.sh apply is the approved apply path (it enforces the saved, guarded plan);
    # strip it before running the raw-apply rules so chained commands are still checked.
    c_raw = re.sub(r"scripts/tf\.sh\s+apply\b", "TFSH_APPLY", c)
    for pat, why in DENY:
        if re.search(pat, c_raw, re.IGNORECASE):
            return "deny", why
    for pat, why in ASK:
        if re.search(pat, c, re.IGNORECASE):
            return "ask", why
    return None, None


def decide_edit(tool, ti):
    path = ti.get("file_path", "")
    if not path.endswith((".tf", ".tf.json")):
        return None, None
    pairs = []
    if tool == "Edit":
        pairs = [(ti.get("old_string", ""), ti.get("new_string", ""))]
    elif tool == "MultiEdit":
        pairs = [(e.get("old_string", ""), e.get("new_string", "")) for e in ti.get("edits", [])]
    for old, new in pairs:
        for m in PROTECT_MARKERS:
            if old.count(m) > new.count(m):
                return "ask", f"edit removes '{m}' in {path} — this is a safety guard, needs human approval"
        if re.search(r'"?Resource"?\s*[=:]\s*\[?\s*"\*"', new) and not re.search(r'"?Resource"?\s*[=:]\s*\[?\s*"\*"', old):
            return "ask", f'edit introduces Resource "*" in {path}'
        if re.search(r"0\.0\.0\.0/0|::/0", new) and not re.search(r"0\.0\.0\.0/0|::/0", old):
            return "ask", f"edit introduces 0.0.0.0/0 in {path} (egress may be fine; ingress is not)"
    if tool == "Write":
        content = ti.get("content", "")
        if re.search(r'Resource"?\s*[=:]\s*\[?\s*"\*"', content):
            return "ask", f'new file content contains Resource "*" ({path})'
    return None, None


def main():
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        return 0
    tool, ti = ev.get("tool_name", ""), ev.get("tool_input", {}) or {}
    if tool == "Bash":
        d, why = decide_bash(ti.get("command", ""))
    elif tool in ("Edit", "MultiEdit", "Write"):
        d, why = decide_edit(tool, ti)
    else:
        d, why = None, None
    if d:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": d,
            "permissionDecisionReason": f"NOVA safety guard: {why}",
        }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
