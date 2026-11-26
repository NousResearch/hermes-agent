# Chrome Dependency Troubleshooting for WSL

## The Error

When Chrome auto-launch fails on a fresh WSL install:

```
Auto-launch failed: Chrome exited early (exit code: 127)
Chrome stderr:
  ... error while loading shared libraries: libnspr4.so: cannot open shared object file
```

## Required Libraries

These are the packages Chrome needs that are typically missing on minimal WSL:

```
libnspr4 libnss3 libatk-bridge2.0-0t64 libgtk-3-0t64 libgbm1
libxkbcommon0 libxshmfence1 libcups2t64 libdrm2 libasound2t64
```

Plus from the full `--with-deps` list:
```
libxcb-shm0 libx11-xcb1 libx11-6 libxcb1 libxext6 libxrandr2
libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxi6
libpangocairo-1.0-0 libpango-1.0-0 libatk1.0-0t64
libcairo-gobject2 libcairo2 libgdk-pixbuf-2.0-0 libxrender1
libfreetype6 libfontconfig1 libdbus-1-3 libdrm2 libxkbcommon0
libatspi2.0-0t64 libxshmfence1
```

## Install Command (requires sudo)

```bash
sudo apt-get update
sudo apt-get install -y libnspr4 libnss3 libatk-bridge2.0-0t64 \
  libgtk-3-0t64 libgbm1 libxkbcommon0 libxshmfence1 \
  libcups2t64 libdrm2 libasound2t64
```

Or use agent-browser's auto-installer:
```bash
agent-browser install --with-deps
```

## Why This Happens

WSL2 minimal images don't ship GTK, X11 libraries, or desktop audio stacks. Chrome needs them even in headless mode because the same binary serves both headed and headless modes — the libraries are linked at compile time.

## Recommended Alternative

Use **Lightpanda** engine instead. It's a pure Zig binary with zero system dependencies. See the main SKILL.md for setup instructions.