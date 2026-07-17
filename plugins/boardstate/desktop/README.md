# Boardstate — desktop app plugin (phase 2)

The Board as a first-class desktop-app page. Same backend as the web plugin
(`../dashboard/`); this is only the Electron-side frontend.

Install (user):
    mkdir -p ~/.hermes/desktop-plugins/boardstate
    cp plugin.js ~/.hermes/desktop-plugins/boardstate/plugin.js

The desktop app hot-loads it; a **Board** entry appears in the sidebar. Requires the
Boardstate dashboard plugin (backend) to be installed + enabled.
