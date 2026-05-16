# Feature Request: Native User Permission Groups Control

## Summary

Implement a first-class user permission groups feature in Hermes Agent, allowing administrators to define user groups with different capability levels. This enables enterprise deployments where different users have different access rights to system operations, tools, and sensitive data.

## Motivation

Currently, permission control in Hermes Agent relies on system prompt engineering (embedding permission rules in the system prompt). This approach has several limitations:

1. **Not scalable**: Permission rules are baked into the prompt, making it hard to manage
2. **Not enforceable**: LLM-based permission enforcement can be bypassed
3. **No dynamic updates**: Changes require modifying config files manually
4. **No audit trail**: No structured logging of permission checks
5. **Platform-specific workarounds**: Different platforms (WeCom, Telegram, Discord) handle permissions differently

## Proposed Solution

### 1. Configuration Schema

Add a new `permission_groups` section in `config.yaml`:

```yaml
permission_groups:
  enabled: true
  default_group: user
  
  groups:
    admin:
      description: "Full system access"
      allowed_tools:
        - terminal
        - file
        - web
        - browser
        - cronjob
        - skills
        - memory
        - mcp
        - delegation
        - code_execution
        - vision
        - image_gen
        - tts
        - session_search
        - todo
      allowed_actions:
        - system_operations
        - config_management
        - skill_management
        - user_management
        - audit_log_access
      tool_quota: null  # unlimited
      bypass_approval: true
      
    operator:
      description: "Operational staff - can run commands but not modify system"
      allowed_tools:
        - terminal
        - file
        - web
        - browser
        - delegation
        - vision
      allowed_actions:
        - execute_commands
        - read_configs
        - view_logs
      tool_quota:
        terminal:
          max_per_hour: 100
        delegation:
          max_per_day: 50
      bypass_approval: false
      
    user:
      description: "Regular users - knowledge QA only"
      allowed_tools:
        - web
        - vision
      allowed_actions:
        - knowledge_qa
        - general_conversation
      tool_quota:
        web:
          max_per_hour: 20
      bypass_approval: false
      
    guest:
      description: "Minimal access - read only"
      allowed_tools:
        - web
      allowed_actions:
        - knowledge_qa
      tool_quota:
        web:
          max_per_hour: 10
      bypass_approval: false

# User to group mapping
user_group_mapping:
  # platform:user_id -> group
  wecom:admin_user_001: admin
  wecom:operator_001: operator
  wecom:user_001: user
  telegram:admin: admin
  discord:guest_user: guest
  cli:root: admin
```

### 2. Core API

```python
# New file: hermes_agent/permission.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PermissionAction(Enum):
    SYSTEM_OPERATIONS = "system_operations"
    CONFIG_MANAGEMENT = "config_management"
    SKILL_MANAGEMENT = "skill_management"
    USER_MANAGEMENT = "user_management"
    AUDIT_LOG_ACCESS = "audit_log_access"
    EXECUTE_COMMANDS = "execute_commands"
    READ_CONFIGS = "read_configs"
    VIEW_LOGS = "view_logs"
    KNOWLEDGE_QA = "knowledge_qa"
    GENERAL_CONVERSATION = "general_conversation"

@dataclass
class UserContext:
    platform: str      # "wecom", "telegram", "discord", "cli"
    user_id: str       # platform-specific user ID
    group: str         # "admin", "operator", "user", "guest"
    is_authenticated: bool
    
class PermissionChecker:
    def __init__(self, config: dict):
        self.config = config
        self.groups = config.get("permission_groups", {}).get("groups", {})
        self.user_mapping = config.get("permission_groups", {}).get("user_group_mapping", {})
        
    def get_user_context(self, platform: str, user_id: str) -> UserContext:
        """Resolve user context from platform and user_id."""
        group = self.user_mapping.get(f"{platform}:{user_id}", 
                                       self.config.get("permission_groups", {}).get("default_group", "user"))
        return UserContext(
            platform=platform,
            user_id=user_id,
            group=group,
            is_authenticated=True
        )
    
    def can_use_tool(self, context: UserContext, tool_name: str) -> bool:
        """Check if user can use a specific tool."""
        group_config = self.groups.get(context.group, {})
        allowed_tools = group_config.get("allowed_tools", [])
        return tool_name in allowed_tools or "all" in allowed_tools
    
    def can_perform_action(self, context: UserContext, action: PermissionAction) -> bool:
        """Check if user can perform a specific action."""
        group_config = self.groups.get(context.group, {})
        allowed_actions = group_config.get("allowed_actions", [])
        return action.value in allowed_actions
    
    def check_tool_quota(self, context: UserContext, tool_name: str) -> tuple[bool, Optional[str]]:
        """Check if user has remaining quota for tool. Returns (allowed, reason)."""
        group_config = self.groups.get(context.group, {})
        quotas = group_config.get("tool_quota", {})
        
        if tool_name not in quotas:
            return True, None
            
        # Implementation depends on quota tracking (Redis, DB, etc.)
        # Return (False, "Quota exceeded: 100 terminal calls per hour")
        return True, None
```

### 3. Integration Points

#### A. Tool Registry (`tools/registry.py`)

```python
# Add permission check before tool dispatch
def dispatch_tool(tool_name: str, args: dict, context: UserContext) -> ToolResult:
    checker = PermissionChecker(get_hermes_config())
    
    if not checker.can_use_tool(context, tool_name):
        return ToolResult.error(
            f"Tool '{tool_name}' is not available for your permission group '{context.group}'"
        )
        
    quota_ok, quota_msg = checker.check_tool_quota(context, tool_name)
    if not quota_ok:
        return ToolResult.error(quota_msg)
        
    # Proceed with tool execution
```

#### B. Gateway (`gateway/run.py`)

```python
# Extract user context from platform-specific message
def extract_user_context(event: MessageEvent) -> UserContext:
    platform = event.platform
    user_id = event.sender.get("user_id") or event.sender.get("id")
    return permission_checker.get_user_context(platform, user_id)

# Check permissions before processing message
async def handle_message(event: MessageEvent):
    context = extract_user_context(event)
    
    if event.is_tool_call:
        if not permission_checker.can_use_tool(context, event.tool_name):
            await event.reply(f"Permission denied: {event.tool_name}")
            return
```

#### C. Config Schema

```python
# hermes_cli/config.py - add to DEFAULT_CONFIG

"permission_groups": {
    "enabled": False,
    "default_group": "user",
    "groups": {
        "admin": {
            "description": "Full system access",
            "allowed_tools": ["all"],
            "allowed_actions": ["all"],
            "tool_quota": None,
            "bypass_approval": True
        },
        "user": {
            "description": "Regular users",
            "allowed_tools": ["web", "vision"],
            "allowed_actions": ["knowledge_qa", "general_conversation"],
            "tool_quota": {},
            "bypass_approval": False
        }
    },
    "user_group_mapping": {}
}
```

### 4. Audit Logging

```python
# Log all permission checks
def audit_permission_check(context: UserContext, action: str, resource: str, 
                           granted: bool, reason: str = None):
    logger.info({
        "event": "permission_check",
        "timestamp": datetime.utcnow().isoformat(),
        "platform": context.platform,
        "user_id": context.user_id,
        "group": context.group,
        "action": action,
        "resource": resource,
        "granted": granted,
        "reason": reason
    })
```

### 5. CLI Commands

```bash
# View current user permission
/hermes permission me

# Admin: List all users and groups
/hermes permission list

# Admin: Assign user to group
/hermes permission set <platform> <user_id> <group>

# Admin: Check specific user access
/hermes permission check <platform> <user_id> <tool_name>
```

## Implementation Roadmap

### Phase 1: Core Framework (MVP)
- [ ] Add `permission_groups` config schema
- [ ] Implement `PermissionChecker` class
- [ ] Add permission checks to tool registry
- [ ] Basic audit logging

### Phase 2: Platform Integration
- [ ] WeCom user context extraction
- [ ] Telegram user context extraction  
- [ ] Discord user context extraction
- [ ] CLI session context

### Phase 3: Advanced Features
- [ ] Rate limiting / quota tracking
- [ ] Time-based access windows
- [ ] IP-based restrictions
- [ ] Multi-factor authentication support

### Phase 4: Admin UI
- [ ] `/permission` CLI commands
- [ ] Web-based admin panel (optional)
- [ ] Permission inheritance patterns

## Backward Compatibility

When `permission_groups.enabled: false` (default), Hermes Agent behaves exactly as before - no breaking changes to existing deployments.

## Alternative Approaches Considered

1. **System Prompt Engineering** (current state): Not scalable, not enforceable
2. **External IAM Integration** (OAuth/OIDC): Adds complexity, external dependency
3. **RBAC with LDAP/Active Directory**: Enterprise-focused, heavier implementation

The proposed solution balances simplicity (config-based) with enterprise-readiness (structured API, audit logging).

## References

- Current workaround: System prompt-based permission rules in `config.yaml` `agent.system_prompt`
- Related: `approvals` system for dangerous command approval
- Related: `command_allowlist` for command-level restrictions

---

**Author**: Kevinyuyj  
**Date**: 2026-04-11  
**Hermes Agent Version**: Latest (main branch)
