// Pre-renderer boot splash markup (#102419).
//
// The macOS first-launch stall this addresses happens BEFORE the main
// renderer paints (Chromium's trust_store_mac walks the entire system
// keychain on Electron boot; on keychains Apple seeded with a duplicate
// com.apple.kerberos.kdc cert the walk blocks the first paint for minutes).
// The main window stays `show: false` until its first themed paint, so the
// user only sees the Dock's frozen "Opening…" with no feedback at all.
//
// This module builds the tiny splash document the MAIN process shows while
// that window is still unpainted. It is intentionally plain HTML/CSS/JS (no
// preload, no app bundle): a data: URL page that paints its own frame long
// before the heavy renderer load, exactly like the update/launch splash
// windows Linear and VS Code ship. The main process rewrites the one-line
// status via executeJavaScript as real boot-progress phases land.
//
// Pure (no Electron import) so the markup is unit-testable under vitest.

export const BOOT_SPLASH_SHOW_AFTER_MS = 5_000
export const BOOT_SPLASH_WATCH_MS = 250

export interface BootSplashMeta {
  version: string
  // Short install-stamp label (e.g. "abc1234 (main)"). Null in dev checkouts
  // that never ran a build — the footer then shows the version only.
  stampLabel: string | null
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

/**
 * Full standalone splash document. The initial status line is overwritten by
 * the main process as soon as the page paints and boot progress advances.
 */
export function buildBootSplashHtml(meta: BootSplashMeta): string {
  const stamp = meta.stampLabel ? ` · ${escapeHtml(meta.stampLabel)}` : ''

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hermes</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    background: #0e1116;
    color: #e8eaed;
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    -webkit-user-select: none;
    cursor: default;
  }
  .splash { width: 100%; padding: 26px 30px; }
  .line { display: flex; align-items: center; gap: 16px; }
  .spinner {
    flex: none;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 3px solid rgba(232, 234, 237, 0.16);
    border-top-color: #4ade80;
    animation: boot-spin 1.1s linear infinite;
  }
  .title { font-size: 15px; font-weight: 600; letter-spacing: 0.01em; }
  #boot-status {
    margin-top: 5px;
    font-size: 12.5px;
    color: #9aa3b2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 400px;
  }
  .meta {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 18px;
    font-size: 11px;
    color: #6f7885;
    white-space: nowrap;
  }
  @keyframes boot-spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) {
    .spinner { animation: none; border-top-color: #4ade80; }
  }
</style>
</head>
<body>
  <div class="splash">
    <div class="line">
      <div class="spinner" aria-hidden="true"></div>
      <div>
        <div class="title">Still booting — please wait</div>
        <div id="boot-status">Starting Hermes Desktop…</div>
      </div>
    </div>
    <div class="meta">
      <span id="boot-meta">Hermes Desktop v${escapeHtml(meta.version)}${stamp}</span>
      <span id="boot-elapsed" aria-hidden="true"></span>
    </div>
  </div>
  <script>
    (function () {
      var startedAt = Date.now()
      var statusEl = document.getElementById('boot-status')
      var elapsedEl = document.getElementById('boot-elapsed')
      // Called from the main process (executeJavaScript) whenever boot
      // progress advances, so the one-liner stays live during a long stall.
      window.__hermesBootStatus = function (message) {
        statusEl.textContent = message
      }
      function pad(n) { return n < 10 ? '0' + n : String(n) }
      function renderElapsed() {
        var s = Math.floor((Date.now() - startedAt) / 1000)
        var h = Math.floor(s / 3600)
        var m = Math.floor((s % 3600) / 60)
        var sec = s % 60
        elapsedEl.textContent = 'elapsed ' + (h > 0 ? h + ':' + pad(m) + ':' + pad(sec) : m + ':' + pad(sec))
      }
      renderElapsed()
      setInterval(renderElapsed, 1000)
    })()
  </script>
</body>
</html>`
}

/**
 * JS snippet that rewrites the splash status line. `message` is our own boot
 * progress text; JSON.stringify keeps quotes/backslashes safe inside the
 * executeJavaScript source regardless of content.
 */
export function bootSplashStatusScript(message: string): string {
  return `window.__hermesBootStatus && window.__hermesBootStatus(${JSON.stringify(message)})`
}
