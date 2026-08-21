# Capture and Sharing

Read this before taking screenshots or screen recordings, extracting text from
the screen, or sharing files with other machines.

## Screenshots

```bash
omarchy screenshot
omarchy capture screenshot region
omarchy capture screenshot windows
omarchy capture screenshot fullscreen save
```

The first argument picks the mode (`smart|region|windows|fullscreen`), the
second what happens (`slurp|copy|save`). `save` skips the annotation editor
and prints the saved path. Screenshots land in the configured Pictures
directory (override with `OMARCHY_SCREENSHOT_DIR`).

## Screen Recording

```bash
omarchy screenrecord --fullscreen
omarchy screenrecord --stop-recording
```

Optional flags: `--with-desktop-audio`, `--with-microphone-audio`,
`--with-webcam`, `--webcam-device=`, `--webcam-size=`, and
`--resolution=<size>`. Without `--fullscreen` a region picker opens first.
Recordings land in Videos (override with `OMARCHY_SCREENRECORD_DIR`). Resize
a live webcam overlay with `omarchy capture webcam resize <smaller|larger|reset|small|medium|large>`.

If recording fails, rerun with `OMARCHY_SCREENRECORD_DEBUG=true` to collect
`/tmp/omarchy-screenrecord.log` for a bug report.

## Text Capture (OCR)

```bash
omarchy capture text
```

## Sharing Files

```bash
omarchy share clipboard
omarchy share file <path...>
omarchy share folder <path>
omarchy tailscale send <machine> <file...>
omarchy tailscale receive [directory]
omarchy transcode <input> [format] [resolution]
```

