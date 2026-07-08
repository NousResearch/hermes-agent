# AgentMail WebSockets

Use WebSockets when a local agent process needs realtime inbound mail. After an
event arrives, use the CLI for inbox, send, reply, label, and attachment work.

## Python

```python
from agentmail import AgentMail, MessageReceivedEvent, Subscribe

client = AgentMail()

with client.websockets.connect() as socket:
    socket.send_subscribe(Subscribe(
        inbox_ids=["agent@agentmail.to"],
        event_types=["message.received"],
    ))
    for event in socket:
        if isinstance(event, MessageReceivedEvent):
            print(event.message.subject, event.message.from_)
```

## Raw

```text
wss://ws.agentmail.to/v0?api_key=$AGENTMAIL_API_KEY
```

EU region:

```text
wss://ws.agentmail.eu/v0?api_key=$AGENTMAIL_API_KEY
```

Subscribe frame:

```json
{ "type": "subscribe", "event_types": ["message.received"], "inbox_ids": ["agent@agentmail.to"] }
```

Omit `inbox_ids` and `pod_ids` for the API key scope.

## Loop Rules

- Dedupe every event by `event_id`.
- Reconnect with backoff and resubscribe after reconnecting.
