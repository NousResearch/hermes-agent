# Omarchy Shell: Bar, Plugins, and Idle

Read this before changing the status bar, notifications, shell plugins, widgets,
or idle/lock behavior.

The bar, notification daemon, settings panel, and overlays run inside one
long-running Quickshell process (`omarchy-shell`).

```
~/.config/omarchy/shell.json
~/.config/omarchy/plugins/<plugin-id>/
$OMARCHY_PATH/config/omarchy/shell.json
```

The shell hot-reloads `shell.json` on save. `idle.screensaver` and
`idle.lock` are seconds since user idle began.

**Commands:** `omarchy restart shell`, `omarchy refresh shell`

## Bar Layout

```bash
omarchy bar move omarchy.clock --section right
```

For layout edits beyond commands, edit
`~/.config/omarchy/shell.json`; it hot-reloads on save.

## Customizing Built-In Plugins and Widgets

Never edit `$OMARCHY_PATH/shell/plugins/`. Clone it into the user plugin
directory instead:

```bash
omarchy plugin clone omarchy.workspaces
```

Cloning switches the bar to the cloned copy, which is yours to edit and
survives updates. Files under `~/.config/omarchy/plugins/` reload
automatically; force a reload with `omarchy-shell shell rescanPlugins`.

## Idle and Lock

Set `idle.screensaver` and `idle.lock` in
`~/.config/omarchy/shell.json`; "lock after ten minutes" means
`idle.lock` is `600`.

