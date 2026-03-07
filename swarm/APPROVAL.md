# Swarm Approval Gates

Human-in-the-loop approval for risky swarm operations.

## Overview

`ApprovalGate` intercepts actions before execution and decides whether they
need human approval based on a configurable `ApprovalPolicy`.

## Quick start

```python
from swarm.approval import ApprovalGate, DEFAULT_POLICY

gate = ApprovalGate()  # uses DEFAULT_POLICY

# Before executing an action
req = gate.check(task, action="bash")
if req.status == "pending":
    gate.request_approval(req)
    # In another thread / UI callback:
    # gate.approve(req.id)
    approved = gate.wait_for_approval(req.id, timeout=120)
```

## ApprovalPolicy fields

| Field | Default | Purpose |
|---|---|---|
| `require_approval_for_tools` | `["terminal", "git_push", "deploy"]` | Tools that always need approval |
| `require_approval_for_roles` | `["merger"]` | Roles that always need approval |
| `spend_threshold_usd` | `1.0` | Approval required when task spend exceeds this |
| `risk_tiers` | see source | Maps tool/action names to risk levels (`low`/`medium`/`high`/`critical`) |
| `auto_approve_low_risk` | `True` | Skip approval for low-risk actions with no other triggers |

## Risk tiers (defaults)

- **low** -- `read`, `glob`, `grep`, `web_search`, `web_fetch`
- **medium** -- `write`, `edit`
- **high** -- `bash`, `terminal`
- **critical** -- `git_push`, `deploy`

## Thread safety

All `ApprovalGate` methods are thread-safe. `wait_for_approval` blocks the
calling thread until `approve` or `deny` is called from another thread (or
until the timeout expires).
