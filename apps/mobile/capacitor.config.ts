import type { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'com.nousresearch.hermesagent.mobile',
  appName: 'Hermes',
  webDir: 'dist',
  // CapacitorHttp routes the bridge's REST calls (login + ws-ticket) through
  // the NATIVE HTTP stack, which bypasses the browser CORS that the gateway
  // locks to localhost — the whole reason a plain PWA can't talk to a remote
  // gateway but this app can. See src/bridge/http.ts.
  plugins: {
    CapacitorHttp: {
      enabled: true,
    },
    // Makes env(safe-area-inset-*) report the REAL system-bar insets on Android
    // edge-to-edge (the system WebView otherwise only reports display cutouts).
    // Used ONLY to push the titlebar below the status bar — see theme-fallback.css.
    // LIGHT = dark bar icons, matching the default light theme.
    SafeArea: {
      // Initial icon styles (before JS runs) for the default light theme: LIGHT =
      // dark icons on a light background. Both bars are re-synced at runtime to the
      // active app theme via SafeArea.setSystemBarsStyle — see native-init.ts. The
      // bar BACKGROUNDS stay transparent so the themed app background shows behind.
      statusBarStyle: 'LIGHT',
      navigationBarStyle: 'LIGHT',
    },
    // Keyboard is used only for its JS events (keyboardWillShow/Hide) and
    // Keyboard.hide() — see mobile-behaviors. The actual resize-above-keyboard is
    // done natively by Chromium on Android edge-to-edge; the SafeArea plugin is
    // patched (patches/@capacitor-community+safe-area) so it doesn't ALSO pad for
    // the IME, which used to double-collapse the viewport (innerHeight 914 → 241).
    Keyboard: {
      resize: 'none',
    },
  },
  server: {
    // R1 DECISION (highest-risk item): the user's gateway is reached over
    // cleartext (LAN / Tailscale `http://…:9119`), and the chat WebSocket is
    // `ws://`. If the WebView's own origin were `https://localhost`, that
    // `ws://` would be blocked as mixed content. Using the `http` scheme makes
    // the app origin `http://localhost`, so cleartext `ws://` to the gateway is
    // allowed. If you later put the gateway behind TLS (`https://`/`wss://`),
    // flip this back to `https` and drop the cleartext allowance in
    // android/app/src/main/res/xml/network_security_config.xml.
    androidScheme: 'http',
    // Same mixed-content rationale as androidScheme — the chat WebSocket is
    // `ws://` to a LAN/Tailscale gateway. An https app origin would block it.
    iosScheme: 'http',
    // Permit cleartext to the LAN/Tailscale gateway. Scope is narrowed in the
    // Android network-security config; this is the Capacitor-level enable.
    cleartext: true,
  },
}

export default config
