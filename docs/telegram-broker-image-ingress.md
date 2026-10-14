# Telegram broker image ingress

The Telegram adapter always downloads an image into the normal Hermes cache first. An optional
`platforms.telegram.extra.broker_image_ingress_command` can then hand that cached file to the GTX
broker for durable storage and task creation. The hook is disabled unless this command is
configured, and any command failure leaves the Hermes cache path in the event so an image is not
silently lost.

Example profile configuration:

```yaml
platforms:
  telegram:
    extra:
      broker_image_ingress_command:
        - /absolute/path/to/gtx-image-ingress
        - --json-stdin
      broker_image_ingress_timeout_seconds: 30
      broker_image_schema: receipt
```

The command is executed without a shell. It receives one JSON object on stdin:

```json
{
  "source_path": "/home/andyfied/.hermes/cache/images/image.jpg",
  "mime_type": "image/jpeg",
  "kind": "photo",
  "chat_id": 123,
  "message_id": 456,
  "user_id": 789,
  "caption": "optional caption",
  "media_group_id": null,
  "schema": "receipt"
}
```

On success it must write JSON to stdout containing an existing absolute durable path, and may
include the scheduler task identifier:

```json
{
  "path": "/mnt/scratch/gtx-images/incoming/task-.../image.jpg",
  "task_id": "task-..."
}
```

The broker owns validation, task creation, atomic staging, metadata, and retention policy. Hermes
only validates that the returned path is absolute and exists before exposing it to the agent. A
non-zero exit, timeout, malformed response, or missing path is logged and falls back to the normal
Hermes cache. Do not point this setting at an interactive or shell-evaluated command.
