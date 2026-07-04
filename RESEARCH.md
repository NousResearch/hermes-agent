# Zoom Chat Bot Integration Research for Hermes Agent

## Executive Summary

**Feasibility:** ✅ **Feasible** - Zoom Team Chat offers a mature API and webhook infrastructure suitable for Hermes Agent integration.

**Complexity Level:** **Medium-High** (similar to Microsoft Teams adapter)

Zoom Chat bot support can be implemented as a platform plugin following the established adapter pattern in Hermes. The implementation would leverage Zoom's Chat API and webhook infrastructure for receiving messages, with OAuth 2.0 authentication for outbound API calls. Key challenges include OAuth token management, message formatting differences, and webhook server requirements.

---

## Zoom Chat Bot API Overview

### Available APIs

Zoom offers several APIs relevant for chat bot development:

| API | Purpose | Use Case |
|-----|---------|----------|
| **Zoom Team Chat API** | Send/receive messages, manage channels | Core messaging functionality |
| **Chat Webhooks** | Real-time message event notifications | Inbound message delivery |
| **User API** | User profile information | Sender identification |
| **Channel API** | Channel/group management | Multi-channel support |

### Key API Endpoints (Zoom Team Chat)

Based on Zoom's API documentation:

1. **Send a Message**
   - `POST /chat/users/{userId}/messages`
   - Send text or markdown messages to users or channels
   - Supports rich formatting (markdown)

2. **List User's Channels**
   - `GET /chat/users/{userId}/channels`
   - Enumerate available channels

3. **Get Channel**
   - `GET /chat/channels/{channelId}`
   - Channel metadata retrieval

4. **Webhook Events**
   - `chat.message.created` - New message received
   - `chat.message.updated` - Message edited
   - `chat.message.deleted` - Message deleted

### Message Types Supported

- **Text messages** - Plain text and markdown
- **Rich formatting** - Markdown supported (bold, italic, links, code blocks)
- **File attachments** - Via file upload API
- **Interactive elements** - Limited compared to Slack/Teams (no native buttons/cards)

### Rate Limits

Zoom enforces rate limits per API endpoint:
- **Per-minute limits**: Vary by endpoint (typically 30-100 requests/minute)
- **Per-day limits**: Applied at the account level
- Implementation should include rate limiting and exponential backoff

---

## Hermes Gateway Architecture Analysis

### Platform Adapter Pattern

Hermes uses a consistent adapter pattern for all messaging platforms:

```
BasePlatformAdapter (gateway/platforms/base.py)
├── Required methods:
│   ├── __init__(self, config: PlatformConfig)
│   ├── connect() -> bool
│   ├── disconnect()
│   ├── send(chat_id, text, ...) -> SendResult
│   ├── send_typing(chat_id)
│   └── get_chat_info(chat_id) -> dict
├── Optional methods:
│   ├── send_document()
│   ├── send_image()
│   ├── send_voice()
│   └── Interactive UX (clarify, approval, etc.)
└── Helper methods:
    ├── build_source() - Create SessionSource objects
    └── handle_message() - Dispatch inbound to gateway
```

### Plugin Architecture

Zoom Chat should be implemented as a **plugin platform** (not built-in) following the pattern in `plugins/platforms/`:

```
plugins/platforms/zoom/
├── __init__.py          # Exports ZoomAdapter, check_requirements
├── adapter.py           # Main ZoomAdapter implementation
├── plugin.yaml          # Plugin manifest with env vars
└── auth.py              # OAuth token management (optional)
```

### Key Integration Points

Based on `ADDING_A_PLATFORM.md`:

1. **Platform Registry** - Self-register via `plugin.yaml`
2. **Config Handling** - `env_enablement_fn` and `apply_yaml_config_fn` hooks
3. **Authorization** - `allowed_users_env` and `allow_all_env` fields
4. **Message Delivery** - `standalone_sender_fn` for cron delivery
5. **System Prompt Hints** - Platform-specific formatting guidance

### Existing Similar Adapters

| Platform | Transport | Auth | Similarity to Zoom |
|----------|-----------|------|-------------------|
| **Teams** | Webhook + Bot Framework | OAuth 2.0 | **High** - Similar enterprise OAuth pattern |
| **Slack** | Socket Mode + Web API | Bot Token | Medium - Different auth, similar features |
| **Discord** | Gateway WebSocket | Bot Token | Low - Different transport |
| **Webhook** | HTTP POST | HMAC Secret | Medium - Webhook pattern |

---

## Implementation Approach

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Hermes Gateway                            │
├─────────────────────────────────────────────────────────────────┤
│  ZoomAdapter (plugins/platforms/zoom/adapter.py)                │
│  ├── _ZoomWebhookServer (aiohttp)                               │
│  │   └── Receives POSTs from Zoom on /webhooks/zoom              │
│  ├── _ZoomAPIClient                                              │
│  │   └── Sends messages via Zoom Chat REST API                  │
│  ├── _TokenManager                                              │
│  │   └── OAuth 2.0 token acquisition & refresh                   │
│  └── MessageTransformer                                          │
│      └── Zoom markdown ↔ Hermes message format                   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ HTTPS webhook
                              │
┌─────────────────────────────────────────────────────────────────┐
│                         Zoom Cloud                               │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  Chat Webhooks  │───▶│ Hermes Gateway  │                     │
│  │  (events)       │    │  (listener)     │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  Chat API       │◀───│ ZoomAdapter    │                     │
│  │  (send msgs)    │    │ (outbound)     │                     │
│  └─────────────────┘    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. ZoomAdapter Class

```python
class ZoomAdapter(BasePlatformAdapter):
    """Zoom Team Chat platform adapter for Hermes Agent."""
    
    platform_name = "zoom"
    platform_enum = Platform.ZOOM  # Dynamic enum via _missing_
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.ZOOM)
        self._client_id = config.extra.get("client_id") or os.getenv("ZOOM_CLIENT_ID")
        self._client_secret = config.extra.get("client_secret") or os.getenv("ZOOM_CLIENT_SECRET")
        self._bot_jid = config.extra.get("bot_jid") or os.getenv("ZOOM_BOT_JID")
        self._verification_token = config.token or os.getenv("ZOOM_VERIFICATION_TOKEN")
        self._webhook_path = "/webhooks/zoom"
        self._port = int(config.extra.get("port", 8645))
        
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start webhook server and verify OAuth credentials."""
        
    async def send(self, chat_id: str, content: str, ...) -> SendResult:
        """Send message via Zoom Chat API."""
```

#### 2. Webhook Server

Following the Teams adapter pattern:

```python
class _ZoomWebhookHandler:
    """Handles Zoom webhook POSTs."""
    
    WEBHOOK_EVENTS = {
        "chat.message.created": "_handle_message_created",
        "chat.message.updated": "_handle_message_updated",
        "chat.message.deleted": "_handle_message_deleted",
    }
    
    async def handle_webhook(self, request: web.Request) -> web.Response:
        # 1. Verify webhook signature/timestamp
        # 2. Parse event type
        # 3. Route to appropriate handler
        # 4. Return 200 ACK
```

#### 3. OAuth Token Manager

```python
class _ZoomTokenManager:
    """Manages Zoom OAuth 2.0 tokens."""
    
    TOKEN_URL = "https://zoom.us/oauth/token"
    
    async def get_server_to_server_token(self) -> str:
        """Acquire token using client credentials flow."""
        # Zoom Server-to-Server OAuth
        # POST /oauth/token with client_id, client_secret, account_id
```

#### 4. Message Transformation

```python
def _transform_zoom_to_hermes(payload: dict) -> MessageEvent:
    """Convert Zoom message payload to Hermes MessageEvent."""
    return MessageEvent(
        message_id=payload.get("id"),
        chat_id=payload.get("channel_id") or payload.get("user_id"),
        sender_id=payload.get("sender", {}).get("id"),
        text=payload.get("message", ""),
        timestamp=payload.get("timestamp"),
        message_type=MessageType.DIRECT if payload.get("is_direct") else MessageType.GROUP,
        # ... platform-specific fields
    )
```

---

## Authentication & Webhook Setup

### Zoom OAuth 2.0 Flow

Zoom supports multiple OAuth flows:

| Flow | Use Case | Hermes Fit |
|------|----------|------------|
| **Server-to-Server OAuth** | Bot/automation apps | **Recommended** for Hermes |
| **User OAuth** | User-authorized actions | Not needed for bot |
| **JWT** | Legacy (deprecated 2023) | Avoid |

#### Server-to-Server OAuth

**Setup Steps:**

1. **Create Server-to-Server OAuth App**
   - Go to [Zoom Marketplace](https://marketplace.zoom.us/)
   - Create "Server-to-Server OAuth" app
   - Note: `client_id`, `client_secret`, `account_id`

2. **Add Chat Bot Features**
   - Enable "Chat" feature in app settings
   - Configure bot display name and description
   - Set webhook event subscriptions

3. **Configure Scopes**
   ```
   Required scopes:
   - chat_message:write - Send messages
   - chat_message:read - Read messages (for webhook validation)
   - chat_channel:read - List channels
   - imchat:write - IM chat messages
   ```

4. **Webhook Configuration**
   - URL: `https://your-server.com/webhooks/zoom`
   - Events: `chat.message.created`, `chat.message.updated`
   - Verification token: Set `ZOOM_VERIFICATION_TOKEN` in Hermes

### Environment Variables

```bash
# Required
ZOOM_CLIENT_ID=your_client_id
ZOOM_CLIENT_SECRET=your_client_secret
ZOOM_ACCOUNT_ID=your_account_id

# Optional
ZOOM_VERIFICATION_TOKEN=webhook_verification_token
ZOOM_ALLOWED_USERS=user@example.com,other@company.com
ZOOM_ALLOW_ALL_USERS=false
ZOOM_HOME_CHANNEL=channel_id_here
ZOOM_PORT=8645
```

### plugin.yaml Manifest

```yaml
name: zoom-platform
label: Zoom Team Chat
kind: platform
version: 1.0.0
description: >
  Zoom Team Chat gateway adapter for Hermes Agent.
  Connects to Zoom via Server-to-Server OAuth and webhooks
  to relay messages between Zoom chats and Hermes.
author: Hermes Contributors
requires_env:
  - name: ZOOM_CLIENT_ID
    description: "Zoom Server-to-Server OAuth client ID"
    prompt: "Zoom Client ID"
    url: "https://marketplace.zoom.us/"
    password: false
  - name: ZOOM_CLIENT_SECRET
    description: "Zoom Server-to-Server OAuth client secret"
    prompt: "Zoom Client Secret"
    url: "https://marketplace.zoom.us/"
    password: true
  - name: ZOOM_ACCOUNT_ID
    description: "Zoom account ID for Server-to-Server OAuth"
    prompt: "Zoom Account ID"
    url: "https://marketplace.zoom.us/"
    password: false
optional_env:
  - name: ZOOM_VERIFICATION_TOKEN
    description: "Webhook verification token (from Zoom app settings)"
    prompt: "Verification Token"
    password: true
  - name: ZOOM_ALLOWED_USERS
    description: "Comma-separated Zoom user emails allowed to talk to the bot"
    prompt: "Allowed users (comma-separated)"
    password: false
  - name: ZOOM_ALLOW_ALL_USERS
    description: "Allow any Zoom user to trigger the bot (dev only)"
    prompt: "Allow all users? (true/false)"
    password: false
  - name: ZOOM_HOME_CHANNEL
    description: "Default channel ID for cron / notification delivery"
    prompt: "Home channel ID"
    password: false
```

---

## Key Technical Considerations

### Similarities with Existing Adapters

| Feature | Zoom | Teams | Slack | Implementation Notes |
|---------|------|-------|-------|---------------------|
| OAuth 2.0 | ✅ Server-to-Server | ✅ Azure AD | ❌ Bot Token | Reuse Teams OAuth pattern |
| Webhook receiver | ✅ Required | ✅ Required | ❌ Socket Mode | Use aiohttp like Teams |
| Markdown support | ✅ Partial | ✅ Full | ✅ mrkdwn | Transform needed |
| Typing indicator | ❌ Not supported | ✅ | ✅ | Graceful degradation |
| File attachments | ✅ File API | ✅ Graph API | ✅ files.upload | Upload via API |
| Thread support | ✅ Threads | ✅ Reply chains | ✅ ts/thread_ts | Map to Zoom parent_id |
| Edit/Delete messages | ✅ API available | ✅ | ✅ | Implement if needed |

### Key Differences

1. **No Native Bot SDK**
   - Zoom doesn't provide a Python SDK comparable to `microsoft-teams-apps` or `slack-bolt`
   - Must build raw HTTP client using `aiohttp` or `httpx`
   - Reference: Teams adapter wraps `microsoft-teams-apps`

2. **Message Formatting**
   - Zoom uses limited markdown (no blockquotes, limited code blocks)
   - Must strip or transform complex formatting from agent responses
   - Consider: Convert Slack mrkdwn → Zoom markdown

3. **Rate Limiting**
   - Zoom has stricter rate limits than Slack
   - Must implement client-side rate limiting (unlike Discord's 429 handling)
   - Recommend: Token bucket with 30 req/min default

4. **Webhook Verification**
   - Zoom uses a verification token (shared secret) for webhook authenticity
   - Different from Slack's signing secret HMAC pattern
   - Must verify `authorization` header or token in payload

5. **No Typing Indicators**
   - Zoom Chat API doesn't support "user is typing" indicators
   - Unlike Teams/Slack, cannot show typing during long responses
   - Alternative: Send "Thinking..." message, then edit/delete

6. **Bot JID Required**
   - Outbound messages require the bot's Zoom JID (user ID format)
   - Must store bot_jid after first successful token acquisition
   - Different from simple bot token pattern

### Challenges

1. **Token Refresh Management**
   - Server-to-Server tokens expire (typically 1 hour)
   - Must implement token caching and automatic refresh
   - Reference: Teams' `MicrosoftGraphTokenProvider`

2. **Channel vs DM Detection**
   - Messages can be to channels or direct messages
   - Must detect type from webhook payload structure
   - Different API endpoints for each

3. **User Identification**
   - Zoom uses email as primary identifier
   - May need to map email ↔ internal user IDs
   - Consider: Use `sender.email` from webhook payload

4. **Webhook Server Requirements**
   - HTTPS required (Zoom doesn't accept HTTP webhooks)
   - Must handle public internet exposure (unlike Socket Mode)
   - Recommend: Reverse proxy with Let's Encrypt

### Recommended Implementation Order

1. **Phase 1: Core Messaging**
   - Webhook server setup
   - OAuth token acquisition
   - Basic send/receive text messages
   - DM and channel support

2. **Phase 2: Platform Features**
   - Message editing/deletion
   - Thread support (reply to message)
   - File attachment handling
   - Markdown transformation

3. **Phase 3: Hermes Integration**
   - Interactive UX (clarify buttons workaround - use message editing)
   - Home channel for cron delivery
   - Allowed users authorization
   - Platform hints for agent

4. **Phase 4: Polish**
   - Rate limiting
   - Retry logic with exponential backoff
   - Comprehensive error handling
   - Tests

---

## Recommended Next Steps

### 1. Proof of Concept (1-2 days)

- [ ] Create `plugins/platforms/zoom/` directory structure
- [ ] Implement minimal `ZoomAdapter` with webhook receiver
- [ ] Implement Server-to-Server OAuth token acquisition
- [ ] Test sending/receiving a single message to/from Zoom

### 2. Full Adapter Implementation (3-5 days)

- [ ] Complete all required `BasePlatformAdapter` methods
- [ ] Implement message transformation (Zoom ↔ Hermes)
- [ ] Add thread/reply support
- [ ] Handle rate limiting
- [ ] Add comprehensive logging and error handling

### 3. Testing & Documentation (2-3 days)

- [ ] Unit tests for message transformation
- [ ] Integration tests with Zoom sandbox
- [ ] Write `plugins/platforms/zoom/README.md`
- [ ] Add Zoom to Hermes documentation

### 4. Submission

- [ ] Create feature branch
- [ ] Submit PR to Hermes Agent repository
- [ ] Address review feedback

---

## Reference Implementation Patterns

### Teams Adapter (Most Similar)

Location: `plugins/platforms/teams/adapter.py`

Key patterns to reuse:
- OAuth token management
- Webhook server with aiohttp
- Platform-specific message transformation
- Home channel for notifications

### Webhook Adapter

Location: `gateway/platforms/webhook.py`

Key patterns to reuse:
- HMAC verification (adapt for Zoom's token)
- Route-based webhook handling
- Cross-platform delivery

### Slack Adapter

Location: `plugins/platforms/slack/adapter.py`

Key patterns to reuse:
- Block kit transformation (as reference for markdown)
- Message deduplication
- Thread context handling

---

## Resources

### Zoom Documentation
- [Zoom Team Chat API](https://developers.zoom.us/docs/team-chat/)
- [Server-to-Server OAuth](https://developers.zoom.us/docs/internal-apps/)
- [Webhooks Overview](https://developers.zoom.us/docs/api/rest/webhook-reference/)
- [Chat API Reference](https://developers.zoom.us/docs/api/rest/reference/zoom-api/methods/#tag/Chat)

### Hermes Documentation
- [Adding a Platform](gateway/platforms/ADDING_A_PLATFORM.md)
- [Platform Registry](gateway/platform_registry.py)
- [Base Adapter](gateway/platforms/base.py)

### Similar Implementations
- [Teams Adapter](plugins/platforms/teams/adapter.py)
- [Slack Adapter](plugins/platforms/slack/adapter.py)
- [Webhook Adapter](gateway/platforms/webhook.py)

---

## Summary

Zoom Chat bot integration is **feasible and recommended** as a plugin platform adapter. The implementation should follow the Teams adapter pattern (webhook server + OAuth) with adaptations for Zoom's specific API requirements. Estimated effort is **1-2 weeks** for a production-ready implementation.

Key takeaways:
- ✅ Use Server-to-Server OAuth (not JWT or User OAuth)
- ✅ Follow Teams adapter architecture (webhook + aiohttp)
- ✅ Implement as plugin in `plugins/platforms/zoom/`
- ⚠️ No native Python SDK - must build HTTP client
- ⚠️ Rate limiting and token refresh required
- ⚠️ HTTPS webhook endpoint required (public URL)