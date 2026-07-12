# Session presence

Hermes clients can publish small active-session presence records so another
client can discover that a session is live somewhere else.

Presence is not a transport and it does not sync secrets. It is a discovery
layer: "session X is active on host Y, last refreshed at T, with optional
private attach hint Z." Clients can use those records to show live sessions,
resume the stored session locally, or hand off to an environment-specific
adapter such as a remote gateway, SSH, tmux, Syncthing, or a private network
mount.

## Storage

By default, records live under:

```text
$HERMES_HOME/session-presence/active/
```

Set `HERMES_SESSION_PRESENCE_DIR` to place records in any other private shared
directory:

```bash
export HERMES_SESSION_PRESENCE_DIR="$HOME/Sync/hermes/session-presence/active"
```

This is useful when multiple devices can see the same folder. The folder should
be private to the user or trusted devices because records can include hostnames,
working directories, model names, and optional attach hints.

## Record Shape

Each JSON record contains fields such as:

```json
{
  "version": 1,
  "session_id": "runtime-session-id",
  "session_key": "stored-session-id",
  "status": "working",
  "title": "Example task",
  "model": "Hermes-4-405B",
  "cwd": "/home/alice/project",
  "source": "tui_gateway",
  "client": "desktop",
  "profile": "default",
  "endpoint": "",
  "host": "workstation",
  "pid": 1234,
  "instance_id": "workstation-1234",
  "updated_at": 1780780000.0,
  "expires_at": 1780780090.0,
  "metadata": {}
}
```

Records expire automatically from readers' point of view. Writers should refresh
records while the session is alive and clear them when the session finalizes.

## Optional Attach Hints

`HERMES_SESSION_PRESENCE_ENDPOINT` may be set by a client or wrapper to publish
a private attach hint. Hermes treats this as data. Generic clients should not
blindly execute endpoint values from a shared directory.

Examples of endpoint values a private adapter could understand:

```text
ssh://workstation/hermes
tmux://workstation/hermes-phone
https://gateway.example.test/hermes
```

## Desktop

Hermes Desktop reads presence through the gateway RPC method
`session.presence_list` and shows visible records in the sidebar's Live section.
Opening a row resumes the stored session when that session is available to the
active backend. Cross-device attach requires the corresponding transport to be
configured, for example a remote gateway or a private SSH/tmux adapter.
