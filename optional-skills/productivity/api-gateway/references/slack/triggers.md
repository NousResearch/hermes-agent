# Slack Trigger Reference

## Event Types

- [`direct_message.received`](#direct_messagereceived)
- [`channel_message.received`](#channel_messagereceived)
- [`reaction.added`](#reactionadded)

---

## `direct_message.received`

### Parameters

- `channel_id` (string, required): The DM conversation ID to scope to (e.g. `D1234567890`). Only messages in this DM conversation will fire the trigger.

### Sample Payload

```json
{
  "context_team_id": "T00000000001",
  "event_id": "Ev08QYNFBP2A",
  "authorizations": [
    {
      "is_enterprise_install": false,
      "team_id": "T00000000001",
      "is_bot": false,
      "user_id": "U00000000001"
    }
  ],
  "api_app_id": "A00000000001",
  "team_id": "T00000000001",
  "event": {
    "client_msg_id": "34046dd3-7296-4828-ada0-a045dcf2e9a4",
    "blocks": [
      {
        "type": "rich_text",
        "block_id": "7JVNO",
        "elements": [
          {
            "type": "rich_text_section",
            "elements": [
              {
                "type": "text",
                "text": "hey"
              }
            ]
          }
        ]
      }
    ],
    "event_ts": "1782330357.455389",
    "channel": "D00000000001",
    "text": "hey",
    "team": "T00000000001",
    "type": "message",
    "channel_type": "im",
    "user": "U00000000002",
    "ts": "1782330357.455389"
  },
  "type": "event_callback",
  "event_context": "4-eyJldCI6Im1lc3NhZ2UiLCJ0aWQiOiJUMDAwMDAwMDAwMDEiLCJhaWQiOiJBMDAwMDAwMDAwMDEiLCJjaWQiOiJEMDAwMDAwMDAwMDEifQ",
  "is_ext_shared_channel": false,
  "event_time": 1782330357,
  "token": "REDACTED"
}
```

---

## `channel_message.received`

### Parameters

- `channel_id` (string, required): The channel ID to scope to (e.g. `C07HD301C9L`). Only messages posted in this channel will fire the trigger.

### Sample Payload

```json
{
  "context_team_id": "T00000000002",
  "event_id": "Ev0BBTTWMTPX",
  "authorizations": [
    {
      "is_enterprise_install": false,
      "team_id": "T00000000002",
      "is_bot": false,
      "user_id": "U00000000003"
    }
  ],
  "api_app_id": "A00000000002",
  "team_id": "T00000000002",
  "event": {
    "client_msg_id": "d1e55470-ba5c-43f4-ab9d-b9e44982e6ef",
    "blocks": [
      {
        "type": "rich_text",
        "block_id": "AyQPM",
        "elements": [
          {
            "type": "rich_text_section",
            "elements": [
              {
                "type": "text",
                "text": "test test"
              }
            ]
          }
        ]
      }
    ],
    "event_ts": "1781914983.110929",
    "channel": "C00000000001",
    "text": "test test",
    "team": "T00000000002",
    "type": "message",
    "channel_type": "channel",
    "user": "U00000000003",
    "ts": "1781914983.110929"
  },
  "type": "event_callback",
  "event_context": "4-eyJldCI6Im1lc3NhZ2UiLCJ0aWQiOiJUMDAwMDAwMDAwMDIiLCJhaWQiOiJBMDAwMDAwMDAwMDIiLCJjaWQiOiJDMDAwMDAwMDAwMDEifQ",
  "is_ext_shared_channel": false,
  "event_time": 1781914983,
  "token": "REDACTED"
}
```

---

## `reaction.added`

### Parameters

- `reaction` (string, required): The emoji name without colons to match (e.g. `robot_face`, `white_check_mark`).
- `channel_id` (string, optional): Scope to a single channel, or leave blank to match all channels.

### Sample Payload

```json
{
  "context_team_id": "T00000000002",
  "event_id": "Ev0BD0L017QR",
  "authorizations": [
    {
      "is_enterprise_install": false,
      "team_id": "T00000000002",
      "is_bot": false,
      "user_id": "U00000000003"
    }
  ],
  "api_app_id": "A00000000002",
  "team_id": "T00000000002",
  "event": {
    "item": {
      "channel": "C00000000001",
      "type": "message",
      "channel_type": "channel",
      "ts": "1781914983.110929"
    },
    "reaction": "+1",
    "type": "reaction_added",
    "user": "U00000000004",
    "item_user": "U00000000003",
    "event_ts": "1782330741.000100"
  },
  "type": "event_callback",
  "event_context": "4-eyJldCI6InJlYWN0aW9uX2FkZGVkIiwidGlkIjoiVDAwMDAwMDAwMDAyIiwiYWlkIjoiQTAwMDAwMDAwMDAyIiwiY2lkIjoiQzAwMDAwMDAwMDAxIn0",
  "is_ext_shared_channel": false,
  "event_time": 1782330741,
  "token": "REDACTED"
}
```