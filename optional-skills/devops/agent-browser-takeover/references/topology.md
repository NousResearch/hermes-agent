# Lab topology (fake addresses)

These numbers are **documentation only**. They are not a real mesh.
Swap them for your WireGuard addresses at install time.

The **owner VNC URL is always a WireGuard IP**. Loopback is only the
raw RFB socket behind websockify — never the viewer.

```
10.13.37.1   agent VPS   wg0     BIND_IP / noVNC
10.13.37.2   owner laptop        VPN client
10.13.37.4   peer box            optional SOCKS origin
127.0.0.1    VPS loopback        x11vnc RFB + Camoufox WS  (not the viewer)

203.0.113.10   fake VPS WAN      (TEST-NET-3)
198.51.100.20  fake peer WAN     (TEST-NET-2)
```

```
owner 10.13.37.2
  └─ browser  http://10.13.37.1:6080/vnc.html     ← WireGuard, this is “VNC”
       └─ websockify  10.13.37.1:6080             (wg0 only)
            └─ x11vnc  127.0.0.1:5900             (loopback RFB only)
                 └─ Xvfb :98
                      └─ headed Camoufox
                           WS  ws://127.0.0.1:9377/camoufox
                           public HTTP(S) optional:
                             socks5://10.13.37.4:1080
                               └─ peer WAN 198.51.100.20
                           (without proxy, VPS WAN 203.0.113.10)
```

## What must never be public

- Owner viewer `:6080` — `10.13.37.1` (or your `BIND_IP`) only. Never `127.0.0.1`, `0.0.0.0`, public NIC, DNS, reverse proxy.
- Raw RFB `:5900` — VPS loopback only. The owner does not connect here.
- Camoufox WS `:9377` — VPS loopback only.
- Peer SOCKS `:1080` — `10.13.37.4` only.

Owner keystrokes stay in noVNC → x11vnc → X → the bot window. They do not enter the agent context.

## Hermes team / no mesh yet

Do **not** bind noVNC to loopback to “make it easier.” That is a different product than this POC.

If the box has no `wg0` yet, add a dummy address that *looks like* wg0, then run the same `install.sh vps` path:

```bash
"$SKILL_DIR/scripts/lab-dummy-iface.sh"          # prints commands
sudo "$SKILL_DIR/scripts/lab-dummy-iface.sh" apply
# BIND_IP=10.13.37.1  — owner URL http://10.13.37.1:6080/vnc.html
```

Reach that URL from another host only after that host can route to `10.13.37.1` (real WireGuard, or a matching address on the reviewer laptop). Same-box curl to `http://10.13.37.1:6080/vnc.html` is fine for a smoke test.
