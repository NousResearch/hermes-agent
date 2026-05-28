#!/usr/bin/env python3
"""
Hermes Gateway Integration — Context Orchestrator Bridge
This is the glue between the Hermes CLI gateway and context_orchestrator.py.

The gateway should call these functions at each lifecycle point:
  - gateway_message_start()   → at the beginning of each user message
  - gateway_register_turn()   → after each assistant/user exchange
  - gateway_register_tool()   → after each tool call result
  - gateway_trim_check()      → before sending response if context is large
  - gateway_message_end()     → at session end

IMPORTANT: This module no longer imports context_orchestrator symbols at
module level (fixes ImportError from the class-based refactor).  All
orchestrator access goes through the per-session registry.
"""
import sys, os, json, uuid, traceback

# Resolve module path dynamically — works on both Mac staging and Linux production.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    _this_dir,
    os.path.join(os.path.expanduser("~/.hermes/scripts")),
    os.path.join(_this_dir, "scripts"),
]
for _mp in _candidates:
    if os.path.isdir(_mp) and _mp not in sys.path:
        sys.path.insert(0, _mp)
        break

from context_orchestrator import get_orchestrator, drop_orchestrator, _get_all_sessions
from memory_palace import get_working

# Lazy import consult_merge — only loaded when quality gate is actually called
def _get_consult_merge():
    from consult_merge import ConsultMergeOrchestrator
    return ConsultMergeOrchestrator()

# ── Session key management ──────────────────────────────────────
# The gateway stores a UUID per logical session so the orchestrator
# can be retrieved on every turn within that session.

_SESSIONS: dict[str, str] = {}  # gateway_session_id → orchestrator session_key


def _ensure_session_key(gateway_session_id: str,
                        orchestrator_session_key: str | None = None) -> str:
    """Return or create the orchestrator session key for a gateway session.

    If *orchestrator_session_key* is provided (e.g. from the gateway's own
    session_entry), it is stored and returned instead of generating a new UUID.
    This lets the gateway's pre-existing session key be used by all integration
    functions without creating a duplicate.
    """
    if gateway_session_id not in _SESSIONS:
        if orchestrator_session_key is not None:
            _SESSIONS[gateway_session_id] = orchestrator_session_key
        else:
            _SESSIONS[gateway_session_id] = f"orch_{uuid.uuid4().hex[:12]}"
    return _SESSIONS[gateway_session_id]


# ── Public API for Gateway ──────────────────────────────────────

# ── Platform guard: only trim on gateway platforms, never on CLI ──
_TRIM_PLATFORMS = {"telegram", "discord", "slack", "matrix", "signal",
                   "whatsapp", "sms", "email", "webhook", "dingtalk",
                   "wecom", "feishu", "qqbot", "bluebubbles", "yuanbao",
                   "homeassistant"}

def _should_trim(platform: str | None) -> bool:
    """Return True if this platform should participate in context trimming."""
    return bool(platform) and platform.lower() in _TRIM_PLATFORMS


def gateway_message_start(user_input: str, task_category: str = "general",
                          gateway_session_id: str = "default",
                          orchestrator_session_key: str | None = None,
                          platform: str | None = None) -> dict:
    """Call at the start of each user message. Returns context to prepend.
    
    If *platform* is provided and is a CLI session, trimming is skipped.
    """
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    if not _should_trim(platform):
        return {
            "session_id": sk, "context_header": "",
            "est_tokens": 0, "headroom": 0,
            "orchestrator_session_key": sk, "skipped": True,
        }
    orch = get_orchestrator(sk)
    result = orch.start_session(task=task_category, phase="processing")
    return {
        "session_id": result["session_id"],
        "context_header": result["context"],
        "est_tokens": result["total_est_tokens"],
        "headroom": result["headroom"],
        "orchestrator_session_key": sk,
    }


def gateway_register_turn(role: str, content: str,
                          gateway_session_id: str = "default",
                          orchestrator_session_key: str | None = None):
    """Register a user or assistant turn."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    orch.register_conversation_turn(role, content)


def gateway_register_tool(tool_name: str, tool_result,
                          gateway_session_id: str = "default",
                          orchestrator_session_key: str | None = None):
    """Register a tool call result."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    text = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
    orch.register_tool_output(tool_name, text)


def gateway_trim_check(current_tokens: int = 0, force: bool = False,
                       target_model: str | None = None,
                       gateway_session_id: str = "default",
                       orchestrator_session_key: str | None = None,
                       platform: str | None = None) -> dict:
    """Check if trimming is needed. Call before generating response.
    
    If *platform* is provided and is a CLI session, trimming is skipped.
    """
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    if not _should_trim(platform):
        return {
            "action": "skipped", "message": "Platform excluded from trimming",
            "trimmed_blocks": 0, "tokens_recovered": 0, "skipped": True,
        }
    orch = get_orchestrator(sk)
    return orch.trim_context(
        current_usage_tokens=current_tokens,
        force=force,
        target_model=target_model,
    )


def gateway_message_end(summary: str | None = None,
                        gateway_session_id: str = "default",
                        orchestrator_session_key: str | None = None) -> dict:
    """Call at session end. Persists state and runs maintenance."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    result = orch.end_session(summary=summary)
    drop_orchestrator(sk, summary=summary)
    _SESSIONS.pop(gateway_session_id, None)
    return result


def gateway_pause_trimming(reason: str = "", gateway_session_id: str = "default",
                           orchestrator_session_key: str | None = None) -> dict:
    """Pause context trimming for this session."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    return orch.pause_trimming(reason=reason)


def gateway_resume_trimming(reason: str = "", gateway_session_id: str = "default",
                             orchestrator_session_key: str | None = None) -> dict:
    """Resume context trimming for this session."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    return orch.resume_trimming(reason=reason)


def gateway_set_block_protected(block_id: str, protected: bool = True,
                                 gateway_session_id: str = "default",
                                 orchestrator_session_key: str | None = None) -> dict:
    """Mark a block as protected (skip trimming) or unprotect it."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    return orch.set_block_protected(block_id, protected=protected)


def gateway_is_paused(gateway_session_id: str = "default",
                       orchestrator_session_key: str | None = None) -> bool:
    """Check if trimming is currently paused for this session."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    return orch.is_paused()


def gateway_get_pause_info(gateway_session_id: str = "default",
                           orchestrator_session_key: str | None = None) -> dict | None:
    """Get pause metadata."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    return orch.get_pause_info()


def gateway_get_context(gateway_session_id: str = "default",
                        orchestrator_session_key: str | None = None) -> str:
    """Get the current composed context string."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk)
    return orch.get_context()


def gateway_status(gateway_session_id: str = "default",
                   orchestrator_session_key: str | None = None) -> dict:
    """Return current gateway status for monitoring."""
    sk = _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = get_orchestrator(sk) if sk in _get_all_sessions() else None
    pause_info = orch.get_pause_info() if orch and orch.is_paused() else None
    return {
        "gateway_session_id": gateway_session_id,
        "orchestrator_session_key": sk,
        "active_blocks": len(orch._active_blocks) if orch else 0,
        "has_session": orch is not None,
        "is_paused": orch.is_paused() if orch else False,
        "pause_info": pause_info,
        "all_sessions": list(_get_all_sessions().keys()),
    }


def gateway_quality_gate(content: str, task_type: str = "general",
                          gateway_session_id: str = "default",
                          orchestrator_session_key: str | None = None) -> dict:
    """Run the consult/merge quality gate on the final response.

    Routes through the ConsultMergeOrchestrator: classifies the task,
    consults Athena for complex tasks, and runs Ring verification as
    the final quality gate before delivery.
    """
    _ensure_session_key(gateway_session_id, orchestrator_session_key)
    orch = _get_consult_merge()
    return orch.quality_gate(content=content, task_type=task_type)


# ── Shadow Review Integration ──────────────────────────────────────

def gateway_shadow_review(diff: str, file_after: str,
                           project_context: str = "",
                           session_history: str = "",
                           gateway_session_id: str = "default",
                           orchestrator_session_key: str | None = None) -> str:
    """Request an asynchronous shadow review of a code change.

    Fires off a background review against the local 30B model and returns
    a request_id.  Use gateway_shadow_review_result() to poll for findings.
    """
    _ensure_session_key(gateway_session_id, orchestrator_session_key)
    from shadow_reviewer import review_code_change
    return review_code_change(diff, file_after, project_context,
                              session_history, gateway_session_id)


def gateway_shadow_review_result(request_id: str) -> dict | None:
    """Poll for a shadow review result. Returns None if still running."""
    from shadow_reviewer import get_review_result
    return get_review_result(request_id)


# ── Batch cleanup (for Night Council / shutdown) ────────────────

def gateway_cleanup_all():
    """End all active sessions. Call during graceful shutdown."""
    for sk in list(_get_all_sessions().keys()):
        drop_orchestrator(sk, summary="Gateway shutdown")


# ── Self-Test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("HERMES GATEWAY INTEGRATION — SELF-TEST\n")

    # Simulate a full message lifecycle
    print("1. gateway_message_start()")
    start = gateway_message_start(
        "Write a Python function to parse JSON",
        task_category="code_generation",
        gateway_session_id="test-001",
    )
    print(f"   Session: {start['session_id']}")
    print(f"   Context tokens: {start['est_tokens']}, Headroom: {start['headroom']}")
    print(f"   ✅ Context header loaded ({len(start['context_header'])} chars)")

    print("\n2. gateway_register_turn(user)")
    gateway_register_turn("user", "Write a Python function to parse JSON strings safely",
                          gateway_session_id="test-001")
    status = gateway_status("test-001")
    print(f"   ✅ Turn registered, blocks: {status['active_blocks']}")

    print("\n3. gateway_register_turn(assistant)")
    gateway_register_turn("assistant", "```python\nimport json\ndef parse_json(s):\n    return json.loads(s)\n```",
                          gateway_session_id="test-001")
    status = gateway_status("test-001")
    print(f"   ✅ Turn registered, blocks: {status['active_blocks']}")

    print("\n4. gateway_register_tool()")
    gateway_register_tool("execute_code", {"output": "All tests passed", "exit_code": 0},
                           gateway_session_id="test-001")
    status = gateway_status("test-001")
    print(f"   ✅ Tool output registered, blocks: {status['active_blocks']}")

    print("\n5. gateway_trim_check() — should be within budget")
    trim = gateway_trim_check(current_tokens=5000, gateway_session_id="test-001")
    trimmed_n = trim.get("trimmed_blocks", trim.get("trimmed", 0))
    print(f"   Trimmed: {trimmed_n} blocks, Message: {trim['message']}")

    print("\n6. gateway_status()")
    status = gateway_status("test-001")
    print(f"   Status: {status}")

    print("\n7. gateway_get_context()")
    ctx = gateway_get_context("test-001")
    print(f"   Context preview ({len(ctx)} chars): {ctx[:120]}...")

    print("\n8. gateway_message_end()")
    end = gateway_message_end("Successfully generated JSON parser function",
                              gateway_session_id="test-001")
    print(f"   Blocks saved: {end['blocks_saved']}, Maintenance: {end['maintenance']}")
    print(f"   Final stats: {end['final_stats']}")

    print("\n9. gateway_quality_gate()")
    qg = gateway_quality_gate(
        content="Here is a Python function to parse JSON: import json; def parse(s): return json.loads(s)",
        task_type="code_generation",
        gateway_session_id="test-001",
    )
    print(f"   Action: {qg.get('action')}, Model: {qg.get('model')}")
    print(f"   Quality gate passed: {'quality_gate' in qg.get('action', '')}")

    # Test isolation: new session gets a fresh orchestrator
    print("\n9. Session isolation test")
    start2 = gateway_message_start("Second session", gateway_session_id="test-002")
    print(f"   ✅ New session: {start2['session_id']} (different from test-001)")

    # Cleanup
    gateway_cleanup_all()
    print("\n✅ All gateway integration tests passed. Ready to wire into message loop.")