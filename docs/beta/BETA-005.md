# BETA-005 — Risk policy and approval gate

Beta classifies read-only inspection as low risk, preparation/configuration as medium, and production changes, restart, deploy, database termination/deletion, firewall, permissions, or external actions as high risk.

High-risk work is blocked before `delegate_task`. `ApprovalGate` routes the request through Hermes' existing human approval mechanism and issues a short-lived receipt bound to a SHA-256 fingerprint of the exact target, action, impact, rollback, and risk. A changed target or action, denial, missing receipt, or expiration fails closed.

Approval events (`requested`, `approved`, `denied`, `authorized`, `blocked`, `expired`) retain the exact operation fingerprint for audit. Beta ignores approval bypass modes for high-risk requests; explicit consent remains mandatory.

## Validation

```bash
python -m pytest -q tests/agent/beta/test_risk.py
```
