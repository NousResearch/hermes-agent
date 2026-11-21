"""Guided publishing of the browser-hosted Desktop for headless hosts.

The 🖥️ Hermes Desktop (browser) row in ``hermes gateway setup`` / ``hermes setup``
provisions the four conditions a remote browser needs
(listener, secure context, reachability, authentication) on top of the gateway's
own webapp listener and the Web Access sign-in machinery (``webapp_access``):

- local network: avahi mDNS identity + Caddy front door + LAN sign-in origin;
- custom domain: DNS A-record verification + Caddy automatic HTTPS + Nous
  sign-in registered for the public callback.

Linux + systemd only (Caddy/avahi); other hosts keep ``hermes webapp`` and the
Desktop app's Web Access pane. See website/docs/user-guide/desktop-publishing.md.
"""