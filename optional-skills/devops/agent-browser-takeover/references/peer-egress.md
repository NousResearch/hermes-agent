# Optional: browser egress via another WireGuard peer

Same shape as a VPS agent plus a home NAS / travel router / second box
on the mesh: the headed browser's *public* HTTP(S) should leave from the
peer, not the VPS WAN. Takeover noVNC stays on the VPS `BIND_IP`.

Do **not** change the VPS default route. Do **not** bind SOCKS on
`0.0.0.0`. Do **not** point Camoufox at a public proxy.

SSH `-D` is not the path. Peers often have `AllowTcpForwarding no`;
the SOCKS server has to run *on the peer*.

## 1. Peer (origin)

On the other WireGuard host:

```bash
mkdir -p ~/.hermes/takeover
# copy templates/env.example → ~/.hermes/takeover/env
# set PEER_WG_IP to *this* host's wg0 IPv4 (not the VPS)
./scripts/install.sh peer
```

That binds `PEER_WG_IP:1080` only and enables user unit `wg-socks5`.

Sanity from the VPS:

```bash
curl -4 -sS --max-time 10 --socks5-hostname PEER_WG_IP:1080 https://api.ipify.org
curl -4 -sS --max-time 10 https://api.ipify.org
# first should match the peer's public origin; second is the VPS WAN
```

## 2. VPS (browser)

In `~/.hermes/takeover/env`:

```
HERMES_BROWSER_PROXY=socks5://PEER_WG_IP:1080
```

Restart `camoufox-server`. `camoufox_server.py` then sets Firefox
`network.proxy.socks_remote_dns=true` and turns WebRTC off so the VPS
WAN is not leaked. Playwright `connect()` to
`ws://127.0.0.1:9377/camoufox` inherits the proxy.

A one-off `Camoufox()` / `AsyncCamoufox()` **must** pass the same
`proxy=` or it bypasses the peer.

Hold `https://api.ipify.org` in the headed tab: it must match the peer
origin, not the VPS WAN. `scripts/verify.sh` checks that when
`HERMES_BROWSER_PROXY` is set.

## 3. What stays local

| Path | Where |
|---|---|
| Playwright WS `:9377` | VPS loopback |
| Raw RFB `:5900` | VPS loopback |
| noVNC `:6080` | VPS `BIND_IP` (VPN) |
| Browser public HTTP(S) | peer SOCKS |

Owner takeover traffic is VPN → VPS noVNC. Only the *page's* internet
origin moves to the peer.
