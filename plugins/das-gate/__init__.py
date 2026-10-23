"""das-gate plugin — Decoupled Authorization for State-changing Actions (DAS).

Closes the four DAS gaps that the gateway-level dangerous-command regex
detector does NOT cover:

  1. Memory writes (MEMORY.md / USER.md / targets other than the conversation
     log). `request_tool_approval` was already exposed (line 4080 in
     tools/approval.py) but no tool actually invokes it for memory writes —
     prompt injection could persist instructions into long-lived memory.

  2. MCP tool calls (mcp__*__*) — anything routed through the MCP client
     bypasses `DANGEROUS_PATTERNS` because those patterns are scoped to the
     terminal tool. An MCP `slack.send_message` or `github.create_issue`
     slips through ungated.

  3. `git push --force` — destroys PR history; the existing regex set covers
     `rm -rf` but not this.

  4. Destructive docker: `docker rm -f`, `docker system prune`, `docker
     rmi -f`, `docker volume prune` — all silently mass-delete resources.

Wired as a `pre_tool_call` plugin hook. For each flagged tool we invoke
`tools.approval.request_tool_approval` which is the SAME human gate that
Tier-2 dangerous shell patterns use: same [o]nce / [s]ession / [a]lways /
[d]eny flow, same session/permanent allowlist, same timeout fail-closed.
Reusing the primitive is the whole point — one approval machinery, not two.

Configuration (env vars, all optional):
  DAS_GATE_OFF=1              Disable the plugin entirely (still load).
  DAS_GATE_MEMORY_OFF=1       Skip the memory-write gate.
  DAS_GATE_MCP_OFF=1          Skip the MCP tool gate.
  DAS_GATE_DANGEROUS_OFF=1    Skip the git/docker gate (forwards to the
                              existing terminal-tool approval path; this
                              plugin is the LAST line of defense, not a
                              replacement).

Ponytail ceiling: we re-parse the `args` dict shape per tool. If a tool's
signature changes upstream, the gate silently lets it through (no match →
no flag). Upgrade path: import the tool's pydantic schema and validate
args against it before deciding.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def _gate_enabled(name: str) -> bool:
    flag = f"DAS_GATE_{name}_OFF"
    if os.environ.get("DAS_GATE_OFF") == "1":
        return False
    return os.environ.get(flag) != "1"


# ─────────────────────────────────────────────────────────────────────────────
# Memory write detection
# ─────────────────────────────────────────────────────────────────────────────

# Hermes exposes the memory tool under the canonical name `memory` in most
# installs; some legacy profiles use `memorize`. We match both. The arg shape
# is `{target: str, content: str | list}` for `memory` calls and the same
# shape for `memorize` (legacy). The `apply_batch` op also writes through
# memory_tool.MemoryStore.apply_batch — matched by presence of `operations`
# in args.
MEMORY_TOOLS = frozenset({"memory", "memorize", "memory_write", "memory_apply_batch"})


def _memory_target_from_args(tool_name: str, args: Any) -> Optional[str]:
    """Extract the target file (memory/USER/skills/etc.) being written."""
    if not isinstance(args, dict):
        return None
    if tool_name == "memory_apply_batch" or "operations" in args:
        ops = args.get("operations")
        if isinstance(ops, list) and ops:
            targets = {str(op.get("target")) for op in ops if isinstance(op, dict) and op.get("target")}
            targets.discard("")
            if targets:
                return ", ".join(sorted(targets))
    return args.get("target") if isinstance(args.get("target"), str) else None


def _memory_should_flag(tool_name: str, args: Any) -> bool:
    if tool_name not in MEMORY_TOOLS:
        return False
    # Reads are safe (load, list, get, search). Writes/replaces/removes are
    # not — we flag anything that has a `content`, `new_content`, or
    # `operations` payload, OR a method name that looks like mutation.
    if not isinstance(args, dict):
        return False
    write_keys = {"content", "new_content", "operations"}
    if any(k in args for k in write_keys):
        return True
    op = (args.get("op") or args.get("operation") or "").lower()
    if op in {"add", "replace", "remove", "append", "upsert", "set", "delete"}:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MCP tool detection
# ─────────────────────────────────────────────────────────────────────────────

# Anything routed through the MCP client. Tool names are namespaced by server,
# so they look like `mcp__<server>__<tool>` or `<server>.<tool>` depending on
# the MCP client version. We match both.
_MCP_RE = re.compile(r"^(mcp__[a-z0-9_]+__[a-z0-9_]+)|([a-z0-9_-]+\.[a-z0-9_]+)$",
                     re.IGNORECASE)


def _is_mcp_tool(tool_name: str) -> bool:
    if not tool_name:
        return False
    return bool(_MCP_RE.match(tool_name)) or tool_name.startswith("mcp__")


# ─────────────────────────────────────────────────────────────────────────────
# Dangerous command detection (git/docker + gaps in terminal regex)
# ─────────────────────────────────────────────────────────────────────────────

# These complement tools/approval.py:DANGEROUS_PATTERNS. We only flag things
# that regex set misses (verified against das_probe.py spot-checks).
DANGEROUS_TOOL_PATTERNS = [
    (re.compile(r"\bgit\s+push\b[^\n]*--force\b", re.IGNORECASE), "git push --force (rewrites remote history)"),
    (re.compile(r"\bgit\s+push\b[^\n]*-f\b", re.IGNORECASE), "git push -f (rewrites remote history)"),
    (re.compile(r"\bdocker\s+system\s+prune\b", re.IGNORECASE), "docker system prune (mass-deletes containers/images/volumes)"),
    (re.compile(r"\bdocker\s+volume\s+prune\b", re.IGNORECASE), "docker volume prune (mass-deletes volumes)"),
    (re.compile(r"\bdocker\s+image\s+prune\b", re.IGNORECASE), "docker image prune (mass-deletes images)"),
    (re.compile(r"\bdocker\s+rm\b[^\n]*-f\b", re.IGNORECASE), "docker rm -f (force-removes containers)"),
    (re.compile(r"\bdocker\s+rmi\b[^\n]*-f\b", re.IGNORECASE), "docker rmi -f (force-removes images)"),
]


def _command_in_args(args: Any) -> Optional[str]:
    """The terminal tool passes the shell command in either `command` or `cmd`."""
    if not isinstance(args, dict):
        return None
    return args.get("command") or args.get("cmd")


def _dangerous_command_should_flag(args: Any) -> Optional[str]:
    cmd = _command_in_args(args)
    if not cmd:
        return None
    for pattern, desc in DANGEROUS_TOOL_PATTERNS:
        if pattern.search(cmd):
            return desc
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Approval request — delegates to the canonical gateway approval primitive
# ─────────────────────────────────────────────────────────────────────────────

def _request_approval(tool_name: str, reason: str, rule_key: str) -> dict[str, Any]:
    """Single chokepoint that calls the existing approval machinery."""
    try:
        from tools.approval import request_tool_approval
    except ImportError as exc:
        # Fail CLOSED. If the approval system is unreachable, do not let the
        # tool run ungoverned.
        logger.error("das-gate: tools.approval unavailable, refusing %s: %s", tool_name, exc)
        return {
            "action": "block",
            "message": (
                f"das-gate could not reach the approval subsystem for '{tool_name}'. "
                "Tool blocked. Fix the import error or set DAS_GATE_OFF=1."
            ),
        }
    result = request_tool_approval(
        tool_name=tool_name,
        reason=reason,
        rule_key=rule_key,
    )
    # Shape normalization: tools/approval returns {"approved": bool, ...};
    # the pre_tool_call hook contract is {"action": "block"|"approve", ...}.
    if result.get("approved"):
        return {"action": "approve", "message": result.get("message")}
    return {"action": "block", "message": result.get("message") or reason}


# ─────────────────────────────────────────────────────────────────────────────
# Hook entry point
# ─────────────────────────────────────────────────────────────────────────────

def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[dict[str, str]]:
    """Intercept tool calls that match one of the four DAS gaps.

    Returns None to pass through (no flag), or {"action": "block"/"approve", ...}
    to gate. Approval comes from the SAME human-in-the-loop machinery that
    `DANGEROUS_PATTERNS` uses, so the UX (Telegram/Discord prompt with
    once/session/always/deny, allowlist persistence, fail-closed timeout) is
    identical to the existing dangerous-command gate.
    """
    if not tool_name:
        return None

    # Gate 1: memory writes
    if _gate_enabled("MEMORY") and _memory_should_flag(tool_name, args):
        target = _memory_target_from_args(tool_name, args) or "unknown"
        return _request_approval(
            tool_name=tool_name,
            reason=f"Memory write to '{target}' persists across sessions. Confirm this is intentional.",
            rule_key=f"das:memory:{target}",
        )

    # Gate 2: MCP tool calls (any non-trivial payload)
    if _gate_enabled("MCP") and _is_mcp_tool(tool_name):
        # Skip read-only MCP verbs: list_*, get_*, search_*, read_*, fetch_*.
        last = tool_name.rsplit("__", 1)[-1].rsplit(".", 1)[-1].lower()
        if not last.startswith(("list_", "get_", "search_", "read_", "fetch_", "describe_", "schema_")):
            return _request_approval(
                tool_name=tool_name,
                reason=f"MCP tool '{tool_name}' has external side effects. Confirm.",
                rule_key=f"das:mcp:{tool_name}",
            )

    # Gate 3: dangerous git/docker patterns missed by the terminal regex set
    if _gate_enabled("DANGEROUS"):
        danger = _dangerous_command_should_flag(args)
        if danger:
            return _request_approval(
                tool_name=tool_name,
                reason=danger,
                rule_key=f"das:cmd:{re.sub(r'[^a-z0-9]+', '-', danger.lower())[:40]}",
            )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plugin registration
# ─────────────────────────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Called by the plugin loader at gateway startup."""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)


# ─────────────────────────────────────────────────────────────────────────────
# Self-check (ponytail: lazy code without its check is unfinished)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Stub a fake hook context (matches PluginManager.register_hook signature)
    class _StubCtx:
        def __init__(self):
            self.hooks = []
        def register_hook(self, name, fn):
            self.hooks.append((name, fn))

    # Build a minimal stub for `request_tool_approval` since it's only
    # importable inside the gateway's hermes-agent runtime, not from __main__.
    import types
    fake_mod = types.ModuleType("tools.approval")
    def _fake_request(tool_name, reason, rule_key=""):
        # Simulate user DENY for the deny test, APPROVE for the rest.
        if "deny" in reason.lower():
            return {"approved": False, "message": f"denied by user: {reason}"}
        return {"approved": True, "message": None}
    fake_mod.request_tool_approval = _fake_request  # type: ignore[attr-defined]
    import sys
    sys.modules["tools.approval"] = fake_mod

    # Smoke-test: build a stub ctx, run all four cases.
    ctx = _StubCtx()
    register(ctx)
    assert len(ctx.hooks) == 1 and ctx.hooks[0][0] == "pre_tool_call", "hook registration broken"
    pre = ctx.hooks[0][1]

    cases = [
        ("memory", {"target": "memory", "content": "remember X"}, "memory write flagged"),
        ("memory", {"target": "memory"}, "memory read passes"),
        ("mcp__slack__send_message", {"channel": "#x", "text": "hi"}, "MCP write flagged"),
        ("mcp__slack__list_channels", {}, "MCP read passes"),
        ("git_push", {"command": "git push origin main --force"}, "git force-push flagged"),
        ("git_push", {"command": "git push origin main"}, "git normal push passes"),
        ("docker_prune", {"command": "docker system prune -a"}, "docker prune flagged"),
        ("docker_list", {"command": "docker ps"}, "docker list passes"),
        ("terminal", {"command": "ls -la"}, "benign terminal passes"),
    ]
    fails = 0
    for tool, args, label in cases:
        out = pre(tool_name=tool, args=args)
        decision = "PASS" if out is None else out["action"].upper()
        print(f"  {decision:5s}  {tool:35s} {label}")
        if decision == "PASS" and "flagged" in label:
            fails += 1
        if decision != "PASS" and "passes" in label:
            fails += 1
    print(f"\n{'OK' if fails == 0 else 'FAIL'}: {fails} mismatches")
    sys.exit(0 if fails == 0 else 1)
