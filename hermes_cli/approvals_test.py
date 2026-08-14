"""``hermes approvals test`` — dry-run approval verdict for a command.

Answers "what would the approval system do with this command?" WITHOUT
running it, prompting anyone, or persisting anything. It composes the REAL
runtime evaluators from ``tools.approval`` in the same order the runtime
guard (``check_all_command_guards``) applies them:

    1. container-skip gate (isolated backends bypass all guards),
    2. hardline blocklist (never bypassable, fires before yolo/off),
    3. sudo-stdin guard (unconditional),
    4. user ``approvals.deny`` rules (fire before yolo/off),
    5. yolo / ``approvals.mode: off`` bypass,
    6. permanent ``command_allowlist``,
    7. dangerous-pattern detection → would ask for approval.

Because the same functions run — including ``_command_detection_variants``'s
normalization/de-obfuscation path — an obfuscated command (``r\\m -rf /``)
gets exactly the verdict its plain form would get at runtime, and the trace
shows the normalized variants that were actually evaluated.

Read-only invariants: the command is never executed, no approval prompt is
raised, nothing is written to config or approval history, no gateway
notification fires.

Exit codes (script-friendly):
    0  allow (would run without a prompt)
    1  usage error
    2  ask-approval (would raise an interactive approval prompt)
    3  deny (hardline blocklist, sudo-stdin guard, or user deny rule)
"""

from __future__ import annotations

import json

EXIT_ALLOW = 0
EXIT_USAGE = 1
EXIT_ASK = 2
EXIT_DENY = 3

_VERDICT_EXIT = {
    "allow": EXIT_ALLOW,
    "ask-approval": EXIT_ASK,
    "hardline-deny": EXIT_DENY,
    "user-deny": EXIT_DENY,
}


def evaluate_command(command: str, env_type: str = "local") -> dict:
    """Return the dry-run verdict for *command* on *env_type*.

    Pure composition of the runtime evaluators — no execution, no prompt,
    no persistence. Returns a dict with ``verdict``, ``exit_code``,
    ``rule`` (matching guard/pattern name or None), ``detail`` (human
    explanation), and ``normalized_variants`` (the trace of normalized /
    de-obfuscated forms the detectors actually evaluated).
    """
    import tools.approval as approval

    # Sync config-persisted "always" patterns so the allowlist check below
    # sees what the runtime would see (load is read-only).
    try:
        approval.load_permanent_allowlist()
    except Exception:
        pass

    variants = list(approval._command_detection_variants(command))

    def result(verdict: str, rule=None, detail: str = "") -> dict:
        return {
            "command": command,
            "env_type": env_type,
            "verdict": verdict,
            "exit_code": _VERDICT_EXIT[verdict],
            "rule": rule,
            "detail": detail,
            "normalized_variants": variants,
        }

    # 1. Isolated container backends skip every guard (runtime parity:
    #    this fires BEFORE the hardline floor in check_all_command_guards).
    if approval._should_skip_container_guards(env_type):
        return result(
            "allow",
            detail=(f"env_type '{env_type}' is an isolated container backend; "
                    "the runtime skips all command guards for it"),
        )

    # 2. Hardline blocklist — never bypassable, even under yolo.
    is_hardline, hardline_desc = approval.detect_hardline_command(command)
    if is_hardline:
        return result(
            "hardline-deny", rule=hardline_desc,
            detail="matches the hardline blocklist (never bypassable, "
                   "blocked even under --yolo / approvals.mode=off)",
        )

    # 3. Sudo stdin guard — unconditional, like the hardline floor.
    is_sudo_guess, sudo_desc = approval._check_sudo_stdin_guard(command)
    if is_sudo_guess:
        return result(
            "hardline-deny", rule=sudo_desc,
            detail="sudo stdin guard (unconditional block)",
        )

    # 4. User-defined approvals.deny rules — fire before yolo/off.
    deny_pattern = approval._match_user_deny_rule(command)
    if deny_pattern is not None:
        return result(
            "user-deny", rule=deny_pattern,
            detail="matches a user-defined approvals.deny rule in "
                   "config.yaml (blocked even under --yolo / mode=off)",
        )

    # 5. Yolo / approvals.mode=off bypass.
    if (approval._YOLO_MODE_FROZEN
            or approval.is_current_session_yolo_enabled()
            or approval._get_approval_mode() == "off"):
        return result(
            "allow",
            detail="approval bypass active (--yolo or approvals.mode: off); "
                   "only hardline/deny rules would block",
        )

    # 6. Permanent command_allowlist.
    if approval._command_matches_permanent_allowlist(command):
        return result(
            "allow",
            detail="matches command_allowlist in config.yaml "
                   "(permanently approved)",
        )

    # 7. Dangerous-pattern detection → would prompt.
    is_dangerous, pattern_key, description = approval.detect_dangerous_command(command)
    if is_dangerous:
        return result(
            "ask-approval", rule=description,
            detail="matches a dangerous-command pattern; the runtime would "
                   f"raise an interactive approval prompt (pattern key: "
                   f"{pattern_key!r})",
        )

    return result("allow", detail="no guard matched; would run without a prompt")


def _render_text(verdict: dict) -> None:
    print(f"command : {verdict['command']}")
    print(f"env-type: {verdict['env_type']}")
    print(f"verdict : {verdict['verdict']}  (exit {verdict['exit_code']})")
    if verdict["rule"]:
        print(f"rule    : {verdict['rule']}")
    if verdict["detail"]:
        print(f"detail  : {verdict['detail']}")
    print("normalized trace (variants the detectors evaluated):")
    for v in verdict["normalized_variants"]:
        print(f"  - {v}")


def approvals_test_command(args) -> int:
    """Handle ``hermes approvals test <command...>``. Returns the exit code."""
    words = list(getattr(args, "command_words", None) or [])
    # argparse REMAINDER keeps a leading "--" separator; it is not part of
    # the command being evaluated.
    if words and words[0] == "--":
        words = words[1:]
    if not words:
        print("usage: hermes approvals test [--env-type TYPE] [--json] -- <command...>")
        return EXIT_USAGE
    command = " ".join(words)
    env_type = getattr(args, "env_type", None) or "local"

    verdict = evaluate_command(command, env_type=env_type)

    if getattr(args, "json", False):
        print(json.dumps(verdict, indent=2))
    else:
        _render_text(verdict)
    return verdict["exit_code"]
