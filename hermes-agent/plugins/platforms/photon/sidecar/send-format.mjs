// Outbound /send builder selection for the Photon sidecar.
//
// spectrumMarkdown() enables data detection (enableDataDetection) in the
// underlying iMessage API, which can 500 on messages containing raw URLs.
// Plain-text URLs are auto-linked by iMessage anyway, so markdown messages
// that contain a URL are routed through the text builder, while URL-free
// markdown keeps native markdown rendering.
//
// This lives in its own module (rather than inline in index.mjs) so tests can
// execute the real decision logic under node instead of grepping source —
// see tests/plugins/platforms/photon/test_url_send_path.py.

const URL_RE = /https?:\/\/[^\s)'"<>]+/i;

/**
 * Decide which spectrum-ts builder the /send handler should use.
 *
 * @param {string} format "markdown" | "text" (already validated by /send)
 * @param {string} text   the outbound message body
 * @returns {"markdown"|"text"}
 */
export function chooseSendFormat(format, text) {
  if (format === "markdown" && !URL_RE.test(String(text))) {
    return "markdown";
  }
  return "text";
}
