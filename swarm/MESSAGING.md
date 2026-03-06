# Swarm Messaging Bus

Agent-to-agent messaging for the Hermes swarm system.

## Types

### `MessageType` (enum)

| Value     | Purpose                        |
|-----------|--------------------------------|
| `result`  | Task/computation result        |
| `request` | Ask another agent to do work   |
| `status`  | Heartbeat / progress update    |
| `error`   | Error report                   |
| `data`    | Arbitrary data transfer        |

### `SwarmMessage` (dataclass)

| Field        | Type                 | Description                                  |
|--------------|----------------------|----------------------------------------------|
| `from_agent` | `str`                | Sender agent id                              |
| `to_agent`   | `str`                | Recipient agent id, or `"broadcast"` for all |
| `msg_type`   | `MessageType`        | Category of the message                      |
| `payload`    | `dict[str, Any]`     | Arbitrary message data                       |
| `id`         | `str`                | Auto-generated unique id                     |
| `timestamp`  | `datetime`           | Auto-generated creation time                 |
| `in_reply_to`| `str | None`         | Optional id of the message being replied to  |

## MessageBus API

```python
from swarm import MessageBus, SwarmMessage, MessageType

bus = MessageBus()
```

### `send(msg: SwarmMessage)`

Deliver a message. If `to_agent` is `"broadcast"`, delivers to every registered agent except the sender.

### `broadcast(from_agent, msg_type, payload) -> SwarmMessage`

Convenience wrapper: builds a broadcast `SwarmMessage` and sends it. Returns the created message.

### `receive(agent_id, block=True, timeout=None) -> Optional[SwarmMessage]`

Pop the next message from an agent's inbox. Blocks by default; returns `None` on timeout or if `block=False` and the inbox is empty.

### `peek(agent_id) -> list[SwarmMessage]`

Non-destructive snapshot of an agent's inbox.

### `subscribe(agent_id, msg_type, callback)`

Register a callback `(SwarmMessage) -> None` that fires whenever a message of the given type arrives for `agent_id`. Callbacks run synchronously inside `send()`, so keep them fast.

### `get_history(agent_id=None) -> list[SwarmMessage]`

Return the full message log. If `agent_id` is provided, filters to messages where the agent is sender, recipient, or a broadcast was sent.

### `clear(agent_id)`

Drain and discard all messages in an agent's inbox.

## Thread Safety

All operations are protected by a `threading.Lock`. Inbox queues are stdlib `queue.Queue` instances (themselves thread-safe), so `receive()` can safely block without holding the bus lock.

## Example

```python
from swarm import MessageBus, SwarmMessage, MessageType

bus = MessageBus()

# Register agents by sending or receiving
bus.broadcast("planner", "status", {"state": "ready"})

# Direct message
msg = SwarmMessage(
    from_agent="planner",
    to_agent="coder",
    msg_type=MessageType.request,
    payload={"task": "implement feature X"},
)
bus.send(msg)

# Receive (blocking with timeout)
reply = bus.receive("coder", timeout=5.0)
```
