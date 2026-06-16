## Context Verification (Three-Tier)

Context verification prevents incorrect replies when internal session state drifts due to race conditions, queue overflows, or ContextVar leaks.

### Quick Config

```yaml
context_verification:
  # Enable context verification
  enabled: true
  
  # strict_mode: replace response with warning when drift is detected
  # non-strict: allow response but log warning
  strict_mode: false
  
  # Platform message history API timeout
  timeout: 5
  
  # Which platforms to verify (list of platform names)
  # Supported: feishu, telegram
  platforms: [feishu, telegram]
```

### How It Works

| Tier | What | When | Cost |
|:----:|------|------|:----:|
| **1** | Verify the triggering message still exists on the platform | Before sending response | 1 API call |
| **2** | FIFO queue + ContextVar snapshot | Every queued message | 0 API calls |
| **3** | Cross-reference pending message_id against platform history | During drain | 1 API call |

### Platform Support

| Platform | fetch_recent_messages | Status |
|----------|:---------------------:|:------:|
| Feishu | `message.get` + `messages.list` | ✅ Full |
| Telegram | `copy_message` probe + `get_updates` scan | ✅ Full |
| Weixin (WeChat) | iLink `getupdates` message scan | ✅ Full |
| DingTalk | Open API `GET /v1.0/im/messages/{id}` | ✅ Single message |
| Slack | Not implemented (low priority) | ❌ Default (empty) |
| Discord | Not implemented (low priority) | ❌ Default (empty) |

### Multi-Agent Protection

Subagents spawned via `delegate_task()` automatically inherit the parent's session context (platform, chat_id, user_id). ContextVars are snapshotted at spawn time and restored in the child's worker thread, preventing drift when the parent processes another message while the child runs.

- **delegate_tool.py**: ContextVar snapshot/restore for subagent threads + `_context` metadata in results
- **api_server.py**: ContextVar snapshot for concurrent HTTP worker threads
