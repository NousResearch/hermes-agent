"""
Skill Pre-check Hook for Hermes Agent.
Listens to `agent:pre_process` and injects skill recommendations.
"""

import sys
import os
import time

# Ensure the SRA Core is importable.
# We must add the PROJECT ROOT (gateway/builtin_hooks) to sys.path, 
# so we can import the core library.
_hook_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_hook_dir)  # gateway/builtin_hooks/

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.advisor import SkillAdvisor

# Global cache to avoid re-indexing on every single message
_advisor_instance = None
_advisor_last_init = 0

def _get_advisor():
    global _advisor_instance, _advisor_last_init
    now = time.time()
    # Re-index every 60 seconds to pick up new skills without restarting Hermes
    if _advisor_instance is None or (now - _advisor_last_init > 60):
        try:
            # 使用 ~/.hermes/hooks/data/ 存储持久化数据（用户特定）
            data_dir = os.path.expanduser("~/.hermes/hooks/data")
            _advisor_instance = SkillAdvisor(
                skills_dir=os.path.expanduser("~/.hermes/skills"),
                data_dir=data_dir
            )
            _advisor_last_init = now
        except Exception as e:
            print(f"[hooks] Failed to init SkillAdvisor: {e}")
            return None
    return _advisor_instance

async def handle(event_type, context):
    """
    Hermes Hook Handler for `agent:pre_process`.
    
    Args:
        event_type: "agent:pre_process"
        context: Dict containing "message" (the user's input).
    
    Returns:
        Dict: {"message_override": "Modified message with skill context"} or None.
    """
    if event_type != "agent:pre_process":
        return None

    message = context.get("message", "")
    
    # ── Fast Pass Filter ─────────────────────────────────────
    # Skip processing for empty messages, slash commands, or very short text.
    # This prevents unnecessary latency for non-skill tasks.
    if not message:
        return None
    if message.startswith("/"):
        return None
    if len(message.strip()) < 4:
        return None
    # ─────────────────────────────────────────────────────────

    advisor = _get_advisor()
    if not advisor:
        return None

    try:
        # Run matching (typically takes ~30-50ms)
        start = time.monotonic()
        result = advisor.recommend(message, top_k=2)
        elapsed = time.monotonic() - start

        recs = result.get("recommendations", [])
        if not recs:
            return None

        # Format context
        lines = []
        lines.append("[System Note: Skill Runtime Advisor Recommendations]")
        lines.append("Based on your input, the following skills are relevant. "
                     "Review them before executing to avoid reinventing the wheel.")
        lines.append("")
        
        for i, r in enumerate(recs):
            flag = "⭐" if i == 0 else "  "
            lines.append(f"{flag} {r['skill']} (Score: {r['score']:.1f}, {r['confidence']} confidence)")
            lines.append(f"   -> {r.get('description', 'No description')[:100]}")
            if r.get('reasons'):
                lines.append(f"   -> Match reasons: {', '.join(r['reasons'][:2])}")
            lines.append("")

        lines.append(f"[SRA Processing: {elapsed*1000:.0f}ms]")
        lines.append("---")

        # Inject into message
        sra_context = "\n".join(lines)
        
        # We prepend it to the message. The LLM will see this as part of the user turn.
        return {
            "message_override": f"{sra_context}\n\n{message}"
        }

    except Exception as e:
        # Fail silently — never block the user's message
        print(f"[hooks] SRA pre-check error: {e}")
        return None
