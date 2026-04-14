# Web Capture Login-Wall Assist Policy Contract (v1)

Date: 2026-04-04
Status: active
Scope: B4 Browsing/Web OS login-wall operator-assist boundary

---

## 1) Purpose

Define a deterministic, bounded contract for login-wall assist handling in web capture so login-gated runs:
- fail closed,
- emit explicit operator actionability,
- and resume through a known command path instead of ad-hoc triage.

This contract governs **policy shape and validation only**. It does not widen runtime authority.

---

## 2) Canonical artifacts

- Runtime JSON contract artifact:
  - `state/continuity/latest/web_capture_login_contract_<domain>.json`
- Operator-readable markdown contract:
  - `state/continuity/latest/web_capture_login_contract_<domain>.md`
- Schema:
  - `docs/ops/schemas/web_capture_login_wall_contract.schema.json`
- Template/example:
  - `docs/ops/templates/web_capture_login_wall_contract.template.json`
- Validator:
  - `scripts/web_capture_login_wall_contract_validator.py`

---

## 3) Contract states

`status` is required and must be one of:
- `open`: login/captcha assist is pending; operator action is required.
- `resolved`: subsequent successful run cleared the login-wall condition.

---

## 4) Required policy invariants

### When `status=open`
- `resume_command` must exist and be non-empty.
- `incident_actionability` must exist with:
  - `status=open`
  - `action_required=true`
  - non-empty `recommended_commands`
- `recommended_commands` must include `resume_command`.
- `recommended_steps` is optional, but when present each step must carry bounded `step_id` + `command`; if present, one step command must match `resume_command`.

### When `status=resolved`
- `resolved_at` must exist.
- If `incident_actionability` exists, it must be:
  - `status=resolved`
  - `action_required=false`

---

## 5) Validation

Policy packet validation command:

```bash
python3 scripts/web_capture_login_wall_contract_validator.py \
  --contract state/continuity/latest/web_capture_login_contract_example.com.json \
  --json
```

Template/example validation command:

```bash
python3 scripts/web_capture_login_wall_contract_validator.py \
  --contract docs/ops/templates/web_capture_login_wall_contract.template.json \
  --json
```

Fail-closed rule: validator failures block policy-claim completion for WEB-02 support slices.

