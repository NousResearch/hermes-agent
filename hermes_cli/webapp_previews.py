"""Credential-free preview assets: private HTML crosses only a parent/frame message."""

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, Response

from hermes_cli.web_server_surface import route_on

# Previews keep the freedom they have under no policy at all (any http(s) library, image,
# fetch, worker or embed); their isolation is the opaque sandbox, not a source list.
_GUEST_SOURCES = (
    "default-src 'none'; "
    "script-src * data: blob: 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval'; "
    "style-src * data: 'unsafe-inline'; img-src * data: blob:; font-src * data:; "
    "media-src * data: blob:; connect-src * data: blob:; worker-src * data: blob:; "
    "frame-src * data: blob: about:; object-src 'none'; base-uri * about:; "
    "frame-ancestors 'self'"
)
PREVIEW_POLICY = f"{_GUEST_SOURCES}; sandbox allow-scripts; form-action 'none'"

# Plugin `SandboxedFrame` data: HTML (the SDK's allowlist, never allow-same-origin): the
# outer iframe's own sandbox attribute narrows this ceiling to what the plugin asked for,
# and a nested frame can never hold a flag its parent lacks.
GUEST_SANDBOX = (
    "allow-downloads allow-forms allow-orientation-lock allow-pointer-lock allow-presentation allow-scripts"
)
GUEST_POLICY = f"{_GUEST_SOURCES}; sandbox {GUEST_SANDBOX}; form-action *"

# D3's CSV/TSV parsers need `new Function` within the opaque preview origin.
# An inner sandbox keeps file previews inert while the shell receives their HTML.
PREVIEW_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body,iframe{width:100%;height:100%;margin:0;border:0;display:block}</style>
</head><body><script>
const frame = document.createElement('iframe');
frame.referrerPolicy = 'no-referrer';
frame.title = 'Preview content';
const transferableTypes = [
  'MessagePort', 'ReadableStream', 'WritableStream', 'TransformStream', 'ImageBitmap',
  'OffscreenCanvas', 'VideoFrame', 'AudioData', 'ArrayBuffer', 'MediaStreamTrack', 'RTCDataChannel'
].map(name => globalThis[name]).filter(type => typeof type === 'function');
// An explicit stack: a deeply nested payload that cloning accepts must not
// overflow the call stack. Children are pushed in reverse to keep their order.
function transfersFor(event) {
  const transfers = new Set(event.ports);
  const seen = new Set();
  const pending = [event.data];
  while (pending.length) {
    const value = pending.pop();
    if (!value || typeof value !== 'object' || seen.has(value)) continue;
    seen.add(value);
    let children = [];
    if (transferableTypes.some(type => value instanceof type)) {
      transfers.add(value);
    } else if (value instanceof Map) {
      value.forEach((item, key) => children.push(key, item));
    } else if (value instanceof Set) {
      children = [...value];
    } else if (value instanceof Error) {
      children = [value.cause, value.errors];
    } else if (Array.isArray(value) || Object.getPrototypeOf(value) === Object.prototype ||
               Object.getPrototypeOf(value) === null) {
      children = Object.values(value);
    }
    for (let index = children.length - 1; index >= 0; index--) pending.push(children[index]);
  }
  return [...transfers];
}
// A missing target (no guest frame yet) fails both posts, so its ports are closed.
function relay(target, event) {
  let transfers = [...event.ports];
  try {
    transfers = transfersFor(event);
    target.postMessage(event.data, '*', transfers);
  } catch {
    try {
      target.postMessage(event.data, '*');
    } catch {
      for (const value of transfers) {
        if (typeof globalThis.MessagePort === 'function' && value instanceof MessagePort) value.close();
      }
    }
  }
}
addEventListener('message', event => {
  if (event.source === parent && parent !== window &&
      event.origin === new URL(location.href).origin &&
      event.data?.type === 'hermes-preview-document' && typeof event.data.html === 'string') {
    const guest = typeof event.data.sandbox === 'string';
    frame.setAttribute('sandbox', guest ? event.data.sandbox : event.data.scripts === true ? 'allow-scripts' : '');
    // A plugin's onLoad means its guest document loaded, not this shell.
    frame.onload = guest ? () => parent.postMessage({type: 'hermes-guest-loaded'}, '*') : null;
    frame.srcdoc = event.data.html;
    document.body.append(frame);
  } else if (event.source === parent && event.origin === new URL(location.href).origin) {
    relay(frame.contentWindow, event);
  } else if (event.source === frame.contentWindow) {
    relay(parent, event);
  }
});
</script></body></html>"""

# An external same-origin loader can run under the blob wrapper's inherited app CSP.
PREVIEW_LOADER = """const frame = document.querySelector('iframe[data-preview-html]');
function send() {
  frame.contentWindow.postMessage({type: 'hermes-preview-document',
    html: frame.dataset.previewHtml, scripts: true}, '*');
}
frame.addEventListener('load', send);
send();
"""


def mount_preview_assets(application: FastAPI) -> None:
    """Public Webapp assets receive authorized content only via parent messages."""
    dependencies = [Depends(route_on("browser_files"))]

    @application.get("/assets/webapp-preview.html", dependencies=dependencies)
    async def preview_document():
        return HTMLResponse(PREVIEW_HTML, headers={
            "Content-Security-Policy": PREVIEW_POLICY,
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        })

    @application.get("/assets/webapp-guest.html", dependencies=dependencies)
    async def guest_document():
        return HTMLResponse(PREVIEW_HTML, headers={
            "Content-Security-Policy": GUEST_POLICY,
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        })

    @application.get("/assets/webapp-preview.js", dependencies=dependencies)
    async def preview_loader():
        return Response(PREVIEW_LOADER, media_type="text/javascript", headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })
