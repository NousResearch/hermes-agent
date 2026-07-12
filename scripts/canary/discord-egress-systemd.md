# Canary Discord egress bootstrap and systemd template

This is a review template only. It does not create users, keys, credentials,
journals, units, or services, and it must not be enabled or started until the
owner-approved canary deployment gate and root preflight both pass.

The executable is `python -I -m scripts.discord_edge_bootstrap`. Normal service
startup opens an existing initialized journal and never creates one. Journal
creation is a separate one-time owner action:

```bash
sudo install -d -o muncho-discord-egress -g muncho-discord-egress -m 0700 \
  /var/lib/muncho-discord-egress
sudo -u muncho-discord-egress \
  /opt/muncho-canary-release/venv/bin/python -I \
  -m scripts.discord_edge_bootstrap \
  --config /etc/muncho/discord-edge.json \
  --bootstrap-journal
```

The command refuses any existing database or initialization-marker path.
Normal startup refuses a missing database or marker. Journal bootstrap must
never be placed in `ExecStartPre`, because restart must not become authority to
replace lost idempotency state.

The root-owned config is strict JSON and contains paths and pinned public-key
IDs, never token or private-key values. Production defaults pin the socket to
`/run/muncho-discord-egress/edge.sock`, the journal to
`/var/lib/muncho-discord-egress/discord-edge-journal.sqlite3`, the gateway unit
to `hermes-cloud-gateway.service`, and the edge unit to
`muncho-discord-egress.service`.

```json
{
  "service": {
    "socket_path": "/run/muncho-discord-egress/edge.sock",
    "gateway_unit": "hermes-cloud-gateway.service",
    "edge_unit": "muncho-discord-egress.service",
    "gateway_uid": 2001,
    "edge_uid": 2003,
    "edge_gid": 2003,
    "connection_timeout_seconds": 20,
    "max_connections": 4
  },
  "keys": {
    "writer_capability_public_key_file": "/etc/muncho/trust/discord-writer-capability-public.pem",
    "writer_capability_public_key_id": "<lowercase-ed25519-public-key-sha256>",
    "edge_receipt_private_key_file": "/etc/muncho/credentials/discord-edge-receipt-private.pem",
    "edge_receipt_public_key_id": "<lowercase-ed25519-public-key-sha256>"
  },
  "discord": {
    "token_file": "/etc/muncho/discord-edge-credentials/bot-token",
    "credentials_directory": "/etc/muncho/discord-edge-credentials",
    "api_timeout_seconds": 5
  },
  "journal": {
    "path": "/var/lib/muncho-discord-egress/discord-edge-journal.sqlite3",
    "busy_timeout_ms": 5000
  },
  "runtime": {
    "max_proof_age_ms": 5000
  }
}
```

Required local ownership before startup:

- config: root-owned, edge-group, exact mode `0440`;
- writer capability public SPKI PEM: root-owned, edge group, `0440`;
- edge receipt private PKCS#8 PEM: edge UID/GID, `0400`;
- bot token credential: edge UID, `0400`, directly inside the dedicated
  root-controlled `/etc/muncho/discord-edge-credentials` directory; the
  directory is not shared with the writer or gateway;
- journal directory: edge UID, exact `0700`; journal and marker: edge UID,
  exact `0600`;
- socket directory: edge UID and not group/world writable.

Every key file must be a regular non-symlink, single-link file. Bootstrap
rechecks owner, group, mode, size, timestamps, device, and inode through the
opened descriptor and path after reading. Writer and edge signing identities
must be distinct and match their pinned IDs. No environment variable or cloud
secret lookup is supported. The token is supplied only through the explicit
edge-owned credential file pinned by the root-owned config.

Review and substitute the immutable release SHA and numeric identities before
using this unit template:

```ini
[Unit]
Description=Muncho privileged Discord public-egress edge (canary)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=muncho-discord-egress
Group=muncho-discord-egress
RuntimeDirectory=muncho-discord-egress
RuntimeDirectoryMode=0750
StateDirectory=muncho-discord-egress
StateDirectoryMode=0700
ExecStart=/opt/muncho-canary-releases/<exact-40-char-sha>/venv/bin/python -I -m scripts.discord_edge_bootstrap --config /etc/muncho/discord-edge.json
Restart=on-failure
RestartSec=5s
UMask=0077

NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
LockPersonality=yes
MemoryDenyWriteExecute=yes
PrivateDevices=yes
PrivateTmp=yes
ProtectClock=yes
ProtectControlGroups=yes
ProtectHome=yes
ProtectHostname=yes
ProtectKernelLogs=yes
ProtectKernelModules=yes
ProtectKernelTunables=yes
ProtectProc=invisible
ProtectSystem=strict
ProcSubset=pid
RemoveIPC=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
SystemCallArchitectures=native

# Discord HTTPS remains reachable; cloud-instance metadata does not.
IPAddressDeny=169.254.169.254/32

ReadOnlyPaths=/opt/muncho-canary-releases/<exact-40-char-sha>
ReadOnlyPaths=/etc/muncho/discord-edge.json
ReadOnlyPaths=/etc/muncho/trust/discord-writer-capability-public.pem
ReadOnlyPaths=/etc/muncho/credentials/discord-edge-receipt-private.pem
ReadOnlyPaths=/etc/muncho/discord-edge-credentials/bot-token
ReadWritePaths=/run/muncho-discord-egress
ReadWritePaths=/var/lib/muncho-discord-egress

[Install]
WantedBy=multi-user.target
```

`PrivateNetwork=yes` is deliberately absent because the edge requires Discord
HTTPS. `IPAddressDeny=169.254.169.254/32` is mandatory and must be verified
from the live unit. The gateway is only a Unix-socket client and never receives
the bot token or the edge receipt private key. The canary gateway must have
only the minimal group access needed to connect to the socket, while the
server still authenticates its exact UID and current systemd `MainPID` for
every request.
