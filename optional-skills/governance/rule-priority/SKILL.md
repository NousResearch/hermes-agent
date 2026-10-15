---
name: rule-priority
description: L0-L3 rule priority governance for Hermes Agent
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [governance, safety, rules]
    category: governance
    requires_toolsets: []
rule_priority: L3
---

# Rule Priority Governance

Enforce L0–L3 rule priority ordering and conflict resolution.

**Priority levels:** L0 (Universal) > L3 (Global) > L1 (Project) > L2 (User). Same level: last-write-wins.

**Hooks:** `pre_llm_call` — injects sorted rules (L0/L3 hard constraints first, then L1/L2 soft rules). `pre_tool_call` — blocks tools matching L3 `tool_block` rules.

**Config (disabled by default):**
```yaml
plugins:
  entries:
    rule_priority:
      enabled: true
      rules:
        - id: no-rm
          priority: 3
          content: Never use rm -rf /
          tool_block: {tool: bash, args: {command: rm -rf /}}
        - id: be-safe
          priority: 0
          content: Always prioritize user safety
```
