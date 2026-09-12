# Hermes agent install

`hermes skills install` **only copies this directory** into `$HERMES_HOME/skills/`.
It does not apt-get packages, fetch Camoufox, or start VNC. That is intentional:
Hermes does not run host setup as a side effect of installing a skill.

The agent (or a human) then bootstraps the host.

## Two layers

| Layer | What | Command |
|---|---|---|
| Skill | Instructions + scripts | `hermes skills install official/devops/agent-browser-takeover` |
| Host | Xvfb, x11vnc, noVNC, Camoufox, user units | `"$SKILL_DIR/scripts/install.sh" check` then `vps` |

`$SKILL_DIR` is the directory `skill_view` reports (parent of `SKILL.md`), usually
`$HERMES_HOME/skills/devops/agent-browser-takeover` after a hub install.

Do **not** treat a successful skill install as takeover-ready.

## Agent procedure

When the user asks to install or enable takeover:

1. `skill_view(name="agent-browser-takeover")`. Follow this file. Do not invent a second browser or bind `0.0.0.0`.
2. Run **check only** (no sudo, no mutation):

   ```bash
   "$SKILL_DIR/scripts/install.sh" check
   ```

3. Read the `MISSING=` line.
   - `wg` / `bind_ip`: stop. This skill does not create a WireGuard mesh. Ask the user for `BIND_IP` (their `wg0` IPv4). Write it to `~/.hermes/takeover/env`. Optional: `hermes config set skills.config.takeover.bind_ip <ip>`.
   - `xvfb` / `x11vnc` / `websockify` / `novnc`: tell the user you need **sudo once** for those packages, then run `install.sh vps`.
   - `camoufox`: `install.sh vps` creates `~/.hermes/takeover/venv` and fetches the browser (~700MB). Reuse that venv if it already exists.
4. After `vps`, run `"$SKILL_DIR/scripts/verify.sh"`. Hand the owner `http://$BIND_IP:6080/vnc.html` only if that passes.
5. Optional peer SOCKS is a **second host**. Do not install `wg-socks5` on the VPS. See `peer-egress.md`.

Loading this skill mid-session (`/skill agent-browser-takeover`) must not apt-get or restart browsers.

## Why not bundle Camoufox/noVNC into Hermes

They are heavy, Linux-only, and need a VPN address. Optional-skills stay lean; the agent
installs host pieces when the user actually wants takeover. `hermes doctor` does not
cover this stack — `install.sh check` does.

## Idempotence

`install.sh vps` skips apt packages that already provide the commands, and skips
`camoufox fetch` when the venv already has a browser binary.
