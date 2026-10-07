import { createHash } from 'node:crypto'

/**
 * Content-Security-Policy for the packaged renderer.
 *
 * The production renderer is a `file://` document (electron/main.ts loads
 * `pathToFileURL(resolveRendererIndex())`), so there are no response headers to
 * attach a policy to — it has to be a `<meta http-equiv>` injected into the built
 * HTML. That is why this lives in a build step and not in index.html: the dev
 * server needs `'unsafe-eval'` and a websocket to Vite, and neither may ever
 * reach a packaged build.
 *
 * Every source below is here because something real breaks without it. The
 * non-obvious ones, so nobody "tidies" them away:
 *
 * - `'wasm-unsafe-eval'` — shiki's Oniguruma engine is instantiated from an
 *   inlined base64 wasm binary. Without it ALL syntax highlighting silently
 *   falls back to plain text (chat code blocks, file preview, diffs).
 * - `blob:` in script-src — the desktop plugin runtime compiles each plugin to a
 *   `text/javascript` Blob and ESM-imports the blob URL, and the SDK shims it
 *   rewrites bare specifiers to are themselves blob modules. This runs at module
 *   scope during boot, so omitting it is a boot-time error toast, not a lazy one.
 * - `hermes-media:` — local audio/video is streamed through a privileged custom
 *   Electron scheme, not file:/data:. A custom scheme is never implied by 'self'
 *   or by a default-src fallback; it must be named literally or every local clip
 *   dies silently.
 * - the Google Fonts pair — the DEFAULT theme injects a stylesheet <link> at
 *   module-evaluation time, so omitting these is a guaranteed violation on every
 *   launch of a default install. They must be granted together: the stylesheet
 *   (googleapis) is useless without the font files (gstatic).
 * - the three embed origins — social embeds inject provider scripts into the
 *   top-level document by design.
 *
 * `worker-src 'none'` is explicit on purpose: the app has no Worker of any kind,
 * and without it worker-src falls back to script-src — which now contains
 * `blob:`, silently granting blob: workers we never want.
 *
 * `frame-ancestors` is deliberately absent: Chromium ignores it (along with
 * `sandbox` and `report-uri`) when a policy is delivered via <meta>, so including
 * it would only log a warning at every boot and read as protection that is not
 * actually there. Anti-framing for the renderer would need an onHeadersReceived
 * header instead.
 *
 * KNOWN TRADE-OFF: an `about:srcdoc` iframe inherits its embedder's policy, so a
 * strict script-src also stops scripts inside the HTML-artifact preview
 * (right-rail/preview-artifact.tsx). Verified by launching Electron 40, not
 * assumed — a `blob:` iframe inherits identically, so there is no cheap
 * exemption. Interactive artifacts therefore render statically inline and must
 * use the existing "Open in browser" action to run. That is the deliberate cost
 * of closing injected-script execution in a renderer that displays
 * model-authored markdown.
 */
const DIRECTIVES = [
  ['default-src', ["'none'"]],
  [
    'script-src',
    [
      "'self'",
      "'wasm-unsafe-eval'",
      'blob:',
      'https://www.instagram.com',
      'https://www.tiktok.com',
      'https://platform.twitter.com'
    ]
  ],
  ['style-src', ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com']],
  ['img-src', ["'self'", 'data:', 'blob:', 'https:', 'hermes-media:']],
  ['media-src', ["'self'", 'data:', 'blob:', 'https:', 'hermes-media:']],
  ['font-src', ["'self'", 'data:', 'https://fonts.gstatic.com']],
  // Deliberately permissive: the user types an arbitrary gateway host/port at
  // first run and in settings, and the RENDERER dials it directly. A fixed
  // allowlist here would break every remote and cloud connection.
  ['connect-src', ["'self'", 'data:', 'blob:', 'http:', 'https:', 'ws:', 'wss:', 'hermes-media:']],
  ['frame-src', ["'self'", 'data:', 'blob:', 'https:']],
  ['worker-src', ["'none'"]],
  ['object-src', ["'none'"]],
  ['base-uri', ["'none'"]],
  ['form-action', ["'none'"]]
]

const INLINE_SCRIPT = /<script(?![^>]*\ssrc=)([^>]*)>([\s\S]*?)<\/script>/gi

/** sha256-… source expression for one inline script body. */
export function inlineScriptHash(body) {
  return `'sha256-${createHash('sha256').update(body, 'utf8').digest('base64')}'`
}

/**
 * Hashes of every inline <script> in the document, in source order.
 *
 * Read by regex rather than a DOM parse on purpose: the hash covers the exact
 * bytes between the tags, so re-serializing the document (which would normalise
 * whitespace) invalidates the very hash we are computing.
 */
export function inlineScriptHashes(html) {
  return [...html.matchAll(INLINE_SCRIPT)].map(match => inlineScriptHash(match[2]))
}

/** The policy string, with the supplied inline-script hashes folded into script-src. */
export function buildRendererCsp(hashes = []) {
  return DIRECTIVES.map(([name, sources]) =>
    `${name} ${(name === 'script-src' ? [...sources, ...hashes] : sources).join(' ')}`
  ).join('; ')
}

/**
 * Splice the policy in as the first thing in <head>.
 *
 * Must precede the boot script it authorises, and must not disturb any other
 * byte of the document (see inlineScriptHashes).
 */
export function injectCspMeta(html) {
  const policy = buildRendererCsp(inlineScriptHashes(html))
  const meta = `<meta http-equiv="Content-Security-Policy" content="${policy}">`
  const head = html.match(/<head[^>]*>/i)

  if (!head) {
    throw new Error('renderer-csp: no <head> in the built HTML; refusing to ship an unprotected renderer')
  }

  const at = head.index + head[0].length

  return `${html.slice(0, at)}\n    ${meta}${html.slice(at)}`
}

/**
 * Vite plugin. Build only — never the dev server, whose HMR needs sources this
 * policy forbids.
 */
export function rendererCsp() {
  return {
    name: 'hermes-renderer-csp',
    apply: 'build',
    transformIndexHtml: {
      order: 'post',
      handler: html => injectCspMeta(html)
    }
  }
}
