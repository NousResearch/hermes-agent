---
name: hermes-hardening-lessons-from-agent-zero
description: Cross-reference Agent Zero CVEs and community findings against Hermes security posture. 8-pattern audit covering MCP injection, path traversal, SSRF, memory timeouts, agent loops, provider validation, patch persistence, and tool args. Produces a structured spec with implementation priorities, risks, and ownership.
version: 1.0.0
platforms: [linux]
metadata:
  hermes:
    tags: [security, hardening, mcp, agent-zero, wesker]
    related_skills: [code-security, native-mcp]
---

# Using the Hermes Hardening Spec

1. When asked to audit Hermes security posture against known Agent Zero vulnerabilities, load this skill first.
2. Research sources to check:
   - Agent Zero CVEs: CVE-2026-30624 (MCP RCE), CVE-2026-4307 (path traversal), CVE-2026-4308 (SSRF)
   - Community: MrTrenchTrucker NVIDIA routing incident, issues #1088, #1493, #1416
   - OX Security MCP supply chain advisory (April 2026)
3. For each pattern evaluate: (a) describe the vulnerability, (b) does Hermes have it?, (c) implementation cost, (d) owner
4. Produce a markdown spec with: Executive summary + 8 pattern sections + priority matrix + implementation paths + risk register
5. Create child kanban tasks for each action item, ordered by priority

Key Hermes config points to check when evaluating:
- approvals.mode: manual
- security.tirith_enabled: true
- tool_loop_guardrails with hard stops
- auxiliary.*.timeout defaults
- memory provider timeouts
- MCP tool validation (_validate_remote_mcp_url)
- fallback_providers base_url list
