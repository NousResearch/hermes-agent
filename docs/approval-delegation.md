# Approval Delegation

## Overview

The approval delegation mechanism allows dangerous command approvals to be routed to designated admins when the user is not an admin. This supports cross-platform delegation (e.g., user on WeChat, admin on Feishu).

## Configuration

Add `delegate_to` under `approvals` in `config.yaml`:

```yaml
approvals:
  mode: smart
  timeout: 60
  delegate_to:
    - platform: feishu
      user_id: "79b3f488"
      chat_id: "oc_96ce9a9d1b5fe90c72c9322f76de5dbf"
```

### Fields

| Field | Description |
|-------|-------------|
| `platform` | Platform identifier (`feishu`, `weixin`, `wecom`, `telegram`, etc.) |
| `user_id` | Admin's user ID on the platform |
| `chat_id` | Chat ID where approval requests will be sent (may differ from `user_id`) |

### Multiple Admins

```yaml
approvals:
  delegate_to:
    - platform: feishu
      user_id: "admin1_feishu_id"
      chat_id: "admin1_chat_id"
    - platform: wecom
      user_id: "admin2_wecom_id"
      chat_id: "admin2_chat_id"
```

## How It Works

1. User triggers a dangerous command (e.g., `rm -rf /tmp/test`)
2. System checks if user is an admin
3. If not admin, approval request is delegated to configured admin(s)
4. Admin receives an interactive card (Feishu) or text message (other platforms)
5. Admin approves/denies via button click or `/approve`/`/deny` command
6. Result is relayed back to the original user's session

## Platform Support

| Platform | Interactive Cards | Text Fallback |
|----------|------------------|---------------|
| Feishu | ✅ Buttons | ✅ `/approve` `/deny` |
| WeChat | ❌ | ✅ `/approve` `/deny` |
| WeCom | ✅ Buttons | ✅ `/approve` `/deny` |
| Telegram | ❌ | ✅ `/approve` `/deny` |

## Translation Keys

The following translation keys are used (in `locales/*.yaml`):

```yaml
approval_delegation:
  waiting_for_admin: "⏳ Approval required. Notifying admin..."
  approval_request: "🔐 Approval Delegation — Dangerous command requires admin approval..."
  no_admin_reachable: "⚠️ Could not reach admin. Approval request will continue in your session."
  approval_expired: "This approval request has expired."
  admin_approved: "✅ Admin approved. Executing..."
  admin_denied: "❌ Admin denied the operation."
  denied_request: "Approval request denied."
```

## Testing

Run tests:

```bash
pytest tests/gateway/approval_delegation/ -v
```

## Architecture

```
gateway/approval_delegation/
├── __init__.py      # Main module: monkey-patches gateway for delegation
├── delegation.py    # Core delegation logic: config, queue, resolution
└── docs/
    └── approval-delegation.md  # This file

tests/gateway/approval_delegation/
├── conftest.py
└── test_delegation.py  # 24 tests
```
