import logging
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.profile_activity_ledger import record_event_if_enabled
from hermes_cli.config import load_config

logger = logging.getLogger(__name__)

class PolicyEngine:
    def __init__(self):
        self.policies = {}
        self._load_policies()

    def _load_policies(self):
        try:
            config = load_config()
            # Look for policies in config.yaml under 'governance.tool_policies'
            # or in a separate policy.yaml in the profile root.
            policies_cfg = (config.get("governance") or {}).get("tool_policies")
            
            if policies_cfg:
                self.policies = policies_cfg
            else:
                # Fallback to policy.yaml in HERMES_HOME
                hermes_home = os.getenv("HERMES_HOME")
                if hermes_home:
                    policy_path = Path(hermes_home) / "policy.yaml"
                    if policy_path.exists():
                        with open(policy_path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            self.policies = data.get("policies", {})
        except Exception as e:
            logger.error(f"Failed to load tool policies: {e}")

    def evaluate(self, tool_name: str, args: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """
        Evaluates a tool call against loaded policies.
        Returns (action, message). 
        Actions: 'block', 'approve', None (allow).
        """
        tool_policy = self.policies.get(tool_name)
        if not tool_policy:
            return None, None

        # Handle different policy formats (list of constraints or single constraint)
        constraints = tool_policy if isinstance(tool_policy, list) else [tool_policy]
        
        for constraint in constraints:
            # Example: forbidden_pattern
            if "forbidden_pattern" in constraint:
                pattern = constraint["forbidden_pattern"]
                # Check if pattern is in any of the args values
                for val in args.values():
                    if isinstance(val, str) and pattern in val:
                        return "block", constraint.get("message", f"Forbidden pattern found in tool {tool_name}")

            # Example: path_restriction
            if "forbidden_prefix" in constraint:
                prefix = constraint["forbidden_prefix"]
                # Check commonly used path arguments
                for path_arg in ["path", "file", "dir"]:
                    val = args.get(path_arg)
                    if isinstance(val, str) and val.startswith(prefix):
                        return "block", constraint.get("message", f"Path {val} is restricted for tool {tool_name}")

        return None, None

# Singleton engine
_engine = PolicyEngine()

def pre_tool_call_handler(
    tool_name: str,
    args: Dict[str, Any],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: list = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Plugin hook for pre_tool_call.
    Logs every check to the Denji audit ledger.
    """
    # 1. Evaluate policy
    action, message = _engine.evaluate(tool_name, args)
    
    # 2. Log to Denji Audit Ledger
    record_event_if_enabled(
        source="denji-policy-enforcer",
        event_type="tool_policy_check",
        actor_profile=kwargs.get("profile_name", "unknown"),
        object_type="tool_call",
        object_id=tool_name,
        summary=f"Policy {action if action else 'allow'} for {tool_name}",
        payload={
            "action": action,
            "message": message,
            "args": args,
            "tool_call_id": tool_call_id,
            "task_id": task_id,
            "decision": "blocked" if action == "block" else "escalated" if action == "approve" else "allowed"
        }
    )
    
    if action:
        return {"action": action, "message": message}
    
    return {"action": "allow"}

def register(ctx):
    # Register the pre_tool_call hook
    ctx.register_hook("pre_tool_call", pre_tool_call_handler)
    logger.info("Denji Policy Enforcer plugin registered successfully.")
