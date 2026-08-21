# Hyprland Configuration

Read this before changing keybindings, monitors, window rules, or any other
Hyprland configuration.

Omarchy configures Hyprland in Lua. User files are loaded after Omarchy's
defaults, so overrides go here:

```
~/.config/hypr/
├── hyprland.lua
├── bindings.lua
├── monitors.lua
├── input.lua
├── looknfeel.lua
├── autostart.lua
├── hyprsunset.conf
└── xdph.conf
```

**Key behaviors (the `.lua` files):**
- Hyprland auto-reloads on config save
- Use `hyprctl reload` to force reload
- After ANY Hyprland Lua config change, validate with `hyprctl reload` followed by `hyprctl configerrors`
- If `hyprctl configerrors` reports errors, address them and rerun validation until clean or a real blocker is identified
- Use `omarchy refresh hyprland` to reset the Lua config files

The two `.conf` files are read by separate processes, so `hyprctl` neither
applies nor validates them:
- `hyprsunset.conf`: apply with `omarchy restart hyprsunset`; reset with `omarchy refresh hyprsunset`
- `xdph.conf`: applies when the portal restarts, e.g. on next login

## Keybindings

Edit `~/.config/hypr/bindings.lua`:

```lua
o.bind("SUPER + SHIFT + R", "SSH", "alacritty -e ssh your-server")
o.bind("SUPER + B", "Browser", { launch = "chromium" })
```

View current bindings: `omarchy menu keybindings --print`.

When rebinding an existing key:
1. Check existing bindings with `omarchy menu keybindings --print`.
2. If already bound, call `hl.unbind(...)` before `o.bind(...)`.
3. Tell the user what the key was previously bound to.

## Display/Monitors

Edit `~/.config/hypr/monitors.lua`:

```lua
hl.monitor({ output = "eDP-1", mode = "1920x1080@60", position = "0x0", scale = 1 })
hl.monitor({ output = "HDMI-A-1", mode = "2560x1440@144", position = "1920x0", scale = 1 })
```

List monitors and modes with `hyprctl monitors all`.

## Window Rules

**CRITICAL: Hyprland window rules syntax changes frequently between versions.**

Before writing ANY window rules, fetch the current documentation from:
https://wiki.hypr.land/Configuring/Basics/Window-Rules/

Do not rely on cached or memorized syntax. Prefer Omarchy's
`o.window(match, rules)` helper and examples in
`$OMARCHY_PATH/default/hypr/windows.lua`.

