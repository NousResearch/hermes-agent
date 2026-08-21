# Themes, Backgrounds, and Fonts

Read this before changing themes, backgrounds, fonts, or theme colors.

## Theme Commands

```bash
omarchy theme list
omarchy theme current
omarchy theme set <name>
omarchy theme bg next
omarchy theme install <url>
```

## Making a New Theme

1. Create a directory under `~/.config/omarchy/themes`.
2. Inspect `/usr/share/omarchy/themes/catppuccin`.
3. Put matching backgrounds in
   `~/.config/omarchy/themes/<name-of-new-theme>/backgrounds/`.
4. Apply it with `omarchy theme set "Name of new theme"`.

Additional user backgrounds go in
`~/.config/omarchy/backgrounds/<theme-slug>/`.

## Customizing a Stock Theme

Never edit stock themes under `/usr/share/omarchy/themes/`. For a small
tweak, create a same-slug user overlay containing only changed files:

```bash
mkdir -p ~/.config/omarchy/themes/catppuccin
cp /usr/share/omarchy/themes/catppuccin/colors.toml ~/.config/omarchy/themes/catppuccin/
omarchy theme set catppuccin
```

For a fully independent variant, copy the stock theme under a new name:

```bash
cp -r /usr/share/omarchy/themes/catppuccin ~/.config/omarchy/themes/catppuccin-custom
omarchy theme set catppuccin-custom
```

## Fonts

```bash
omarchy font list
omarchy font current
omarchy font set <name>
```

