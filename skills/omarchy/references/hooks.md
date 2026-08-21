# Automation Hooks

Read this before setting up scripts that run on system events (theme changes,
updates, boot, low battery, etc.).

Hooks live in `~/.config/omarchy/hooks/<name>.d/` - one directory per event,
holding any number of independent scripts. Install with
`omarchy hook install <name> <script>`. The runner also executes a flat
`~/.config/omarchy/hooks/<name>` file first, if one exists.

```
~/.config/omarchy/hooks/
├── battery-low.d/          # Low battery (percentage in $1)
├── font-set.d/             # After font change (font name in $1)
├── post-boot.d/            # After the desktop starts
├── post-update.d/          # During omarchy update
├── pre-refresh-pacman.d/   # Before pacman re-syncs packages
└── theme-set.d/            # After theme change (theme slug in $1)
```

Example hook script:

```bash
#!/bin/bash
THEME_NAME=$1
echo "Theme changed to: $THEME_NAME"
# Add custom actions here
```

