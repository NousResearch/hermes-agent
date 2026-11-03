# Installing Hermes Agent on Android (Termux)

> **Audience:** Anyone who wants a fully local Hermes Agent on an Android phone
> **Source files:** `pyproject.toml` (`[termux]` extras), `constraints-termux.txt`, `tools/termux_install.sh`
> **Related:** [Profile Routing](profile-routing.md), [Session Lifecycle](session-lifecycle.md)
> **Tested on:** Samsung Galaxy S25 (SM-S921E), Android 16, Termux from F-Droid/GitHub

## Overview

Hermes Agent runs natively on Android inside [Termux](https://termux.dev) — no root, no
remote server, no PC required. The phone becomes a full Hermes host: local model access
(any OpenAI-compatible provider), the gateway (Telegram/Discord/Slack bots), cron jobs,
and — via [Termux:API](https://github.com/termux/termux-api) and
[Shizuku](https://shizuku.rikka.app) — control of the phone itself (brightness, volume,
SMS, sensors, `input`/`screencap`/`settings` at adb-shell privilege level).

Python wheels for Android are rare (many packages have no `android` platform wheels), so
the install compiles from source inside Termux. The steps below encode the known-good
combination of packages, pins, and environment fixes.

## 1. Install Termux

Install **Termux** from [F-Droid](https://f-droid.org/en/packages/com.termux/) or the
[GitHub releases](https://github.com/termux/termux-app/releases) page (the Play Store
build is deprecated and broken).

```sh
pkg update -y && pkg upgrade -y
pkg install -y python python-pip clang pkg-config libffi openssl rust \
  binutils libc++ git curl
```

> **Pitfall:** `pkg upgrade` may prompt about dpkg conffile changes — answer `y`.

## 2. Bootstrap environment

`TMPDIR` must point inside the Termux prefix or `uvloop`/`cryptography` builds fail with
weird linker errors. Put this in `~/.bashrc` (or `~/env.sh` and source it):

```sh
export PREFIX=/data/data/com.termux/files/usr
export HOME=/data/data/com.termux/files/home
export LD_LIBRARY_PATH=$PREFIX/lib
export PATH=$PREFIX/bin:/system/bin
export TMPDIR=$PREFIX/tmp
```

## 3. Install Hermes

```sh
git clone https://github.com/NousResearch/hermes-agent.git ~/hermes-agent
cd ~/hermes-agent
python -m venv venv
. venv/bin/activate
pip install --upgrade pip wheel
# The [termux] extra pins the Telegram webhooks + cron/cli/mcp/honcho/acp stack.
UVLOOP_USE_SYSTEM_LIBUV=1 pip install -e '.[termux]' -c constraints-termux.txt
```

- `UVLOOP_USE_SYSTEM_LIBUV=1` makes uvloop link against Termux's system libuv instead of
  building a broken bundled one.
- If `cryptography` fails with `Text file busy (os error 26)`, retry — it is a transient
  lock; on repeated failure pin `cryptography==48.0.1` and retry.
- The editable install finishes on a re-run once every wheel is cached.

Then symlink the CLI so it is on `PATH`:

```sh
ln -sf ~/hermes-agent/venv/bin/hermes $PREFIX/bin/hermes
```

## 4. Configure providers & gateway

Copy your model provider keys into `~/.hermes/.env` (chmod 600). Any OpenAI-compatible
endpoint works; example (`config.yaml`):

```yaml
model:
  default: deepseek-v4-flash-free
  provider: opencode-zen
```

Then start the gateway:

```sh
hermes gateway run
```

Pair new platforms (`hermes pairing approve telegram <CODE>`) and lock the bot down with
`TELEGRAM_ALLOWED_USERS` in `~/.hermes/.env`.

## 5. Phone control (optional)

- **Termux:API** — `pkg install termux-api` plus the companion app; grants brightness,
  volume, torch, vibration, camera, mic, sensors, battery, SMS, clipboard, notifications.
- **Shizuku** — grants adb-shell-level powers (`input`, `screencap`, `settings`, `pm`,
  `am`, `dumpsys`) without root. Install the app, start the daemon, then:

  ```sh
  pkg install -y termux-api   # rish comes from the Shizuku app assets
  ```

  Copy `rish` + `rish_shizuku.dex` from the Shizuku APK assets into `$PREFIX/bin`, and add
  `unset LD_LIBRARY_PATH` as line 2 of `rish` (Termux's `LD_LIBRARY_PATH` breaks
  `app_process`). Test with `rish -c id` → `uid=2000(shell)`.

## 6. Battery & lifecycle

- **Termux:Boot** — install the app, then any script in `~/.termux/boot/` runs at boot:
  `nohup hermes gateway run >> ~/.hermes/gateway.log 2>&1 &`
- Whitelist Termux from battery optimization: `adb shell cmd deviceidle whitelist +com.termux`
- The gateway dies with the Termux process — see `tools/termux_install.sh` for the
  on/off helpers.

## Known limitations

- No Android wheels for uvloop/cryptography yet — they build from source (slow first install).
- The Shizuku daemon does not survive a reboot (restart from the app or via adb).
- Some heavy extras (`google`, `homeassistant`, `web`, `pty`) lazy-install on first use.

## Contributing Android support

Android is an officially-supported target in the contribution priorities. PRs welcome:
bug fixes to the `[termux]` extras, the install script, and this guide.
