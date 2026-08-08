# Hermes WhatsApp bridge

The personal-account WhatsApp bridge is launched and authenticated by
`hermes gateway`. Its HTTP semantics run over a private Unix-domain socket;
`bridge_port` remains only a stable instance and multiplex discriminator for
configuration compatibility. It does not open that TCP port.

Do not start serving mode with `node bridge.js`, hand-author
`WHATSAPP_BRIDGE_TOKEN`, or hand-author `WHATSAPP_BRIDGE_ENDPOINT`. Those are
gateway-owned credentials and endpoint state. `npm start` prints this guidance
instead of starting an unauthenticated or incomplete server.

For pairing, run `hermes whatsapp`. Pair-only mode starts no bridge control
server. When developing the package directly, the low-level equivalent is
`node bridge.js --pair-only --session <path>`; always provide the intended
session directory explicitly.

Gateway serving is supported on Linux and macOS. Native Windows named-pipe
runtimes used by Node and aiohttp do not currently provide the pre-write peer
authentication required to keep bearer tokens and message payloads from a
replacement local pipe server. Native Windows therefore fails closed; use
WSL2/Linux for WhatsApp gateway operation. Pair-only mode remains available on
native Windows because it exposes no control server.
