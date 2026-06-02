Snap packaging notes for Hermes Agent.

This package targets strict confinement on Ubuntu and deliberately excludes the
desktop GUI. Both the classic CLI and the `--tui` terminal UI are supported: the
`tui` part builds the Node bundle and stages a Node >=20 runtime inside the snap
(core24's apt `nodejs` is v18, which the TUI's `node20` target rejects), so no
host Node install is needed. The snap entrypoints run through `hermes-snap`, which
sets:

- `HERMES_HOME=$SNAP_USER_COMMON/hermes`
- `HERMES_MANAGED=snap`
- `HERMES_DISABLE_LAZY_INSTALLS=1`
- `HERMES_NODE=$SNAP/bin/node` (the bundled Node runtime, for `--tui`)
- bundled skills and optional skills paths under `$SNAP/usr/share/hermes-agent/`

Build locally from the repository root:

```bash
snapcraft pack
sudo snap install --dangerous hermes-agent_*.snap
```

Smoke-test after install:

```bash
hermes-agent.hermes version
hermes-agent.hermes doctor
hermes-agent.hermes --tui      # launches the terminal UI (uses the bundled Node)
snap start hermes-agent.gateway
snap stop hermes-agent.gateway
```

Command support notes:

- Supported: `chat`, `model`, `fallback`, `secrets`, `migrate`, `gateway run`,
  `proxy`, `lsp`, `setup`, `slack`, `send`, `login`, `logout`, `auth`,
  `status`, `cron`, `webhook`, `portal`, `kanban`, `hooks`, `doctor`,
  `security`, `dump`, `debug`, `backup`, `checkpoints`, `import`, `config`,
  `pairing`, `skills`, `bundles`, `plugins`, `curator`, `memory`, `tools`,
  `mcp`, `sessions`, `insights`, `claw`, `version`, `acp`, `profile`,
  `completion`, `dashboard`, and `logs`.
- Snap-managed alternatives: `update` prints `snap refresh hermes-agent`,
  `uninstall` prints `snap remove hermes-agent`, `postinstall` is a no-op with
  setup guidance, and `gateway start|stop|restart|status` point at `snap`
  service commands.
- Not targeted by this snap: `desktop` / GUI app packaging and macOS-only
  `computer-use`.

Strict confinement means host access is mediated by interfaces. The manifest
uses `home`, `network`, and `network-bind` for the core CLI/gateway/dashboard
flow. `removable-media` is declared for user workspaces under `/media`, `/mnt`,
and `/run/media`, but users may need to connect it manually.
