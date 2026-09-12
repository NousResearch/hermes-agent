# Lab topology (fake addresses)

These numbers are **documentation only**. They are not a real mesh.
Swap them for your WireGuard addresses at install time.

```
10.13.37.1   agent VPS   wg0     BIND_IP
10.13.37.2   owner laptop        VPN client
10.13.37.4   peer box            NAS-shaped SOCKS origin
127.0.0.1    VPS loopback        RFB + Camoufox WS

203.0.113.10   fake VPS WAN      (TEST-NET-3)
198.51.100.20  fake peer WAN     (TEST-NET-2)
```

```
owner 10.13.37.2
  └─ browser  http://10.13.37.1:6080/vnc.html
       └─ websockify  10.13.37.1:6080  (wg0 only)
            └─ x11vnc  127.0.0.1:5900  (loopback only)
                 └─ Xvfb :98
                      └─ headed Camoufox
                           WS  ws://127.0.0.1:9377/camoufox
                           public HTTP(S) optional:
                             socks5://10.13.37.4:1080
                               └─ peer WAN 198.51.100.20
                           (without proxy, VPS WAN 203.0.113.10)
```

## What must never be public

- Raw RFB `:5900` — VPS loopback only
- Camoufox WS `:9377` — VPS loopback only
- noVNC `:6080` — `10.13.37.1` only, never `0.0.0.0` / public NIC / DNS / reverse proxy
- Peer SOCKS `:1080` — `10.13.37.4` only

Owner keystrokes stay in noVNC → x11vnc → X → the bot window. They do not enter the agent context.

## Local reviewer pod (no WireGuard)

`templates/lab/docker-compose.yml` publishes **host** `127.0.0.1:6080` only.
Inside the container the viewer may listen on all interfaces; the host publish
is the trust boundary. Open http://127.0.0.1:6080/vnc.html on the same machine.
