# Hermes systemd units

This directory stores the service definitions used by the Hermes pilot.

The active services currently run as user-level systemd units under
`~/.config/systemd/user/` because this host blocks `sudo` service installation
with `no_new_privileges`.

Use:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
systemctl --user status hermes-dashboard hermes-gateway hermes-nginx hermes-cloudflared
```

All Hermes agent commands run with `--yolo` / approval bypass for the pilot.
