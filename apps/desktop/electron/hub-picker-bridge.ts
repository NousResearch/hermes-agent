/**
 * The picker open-bridge: the ONE way a link inside an embedded hub frame
 * reaches the OS browser.
 *
 * Why not the window-open handler (#135315): `setWindowOpenHandler` is the only
 * place an embedded frame's `window.open` / `target="_blank"` can be observed,
 * and it cannot prove WHICH frame asked. Electron 40's `HandlerDetails` carries
 * no frame, and `webContents.getFocusedFrame()` is `null` for a cross-origin
 * guest even while the window holds focus and the click is a real trusted
 * gesture (measured 2026-10-14 against the live catalog page) — so the
 * focused-frame delegation in window-open-policy.ts never fires for an embed,
 * and `sandbox="... allow-popups ..."` alone only turns a renderer-side silent
 * block into a handler-side denial.
 *
 * Widening the handler instead is NOT an option: `isHermesHubExternalUrl` is a
 * scheme allowlist, so "a hub frame exists in this window" as the delegation
 * gate would let a sandboxed artifact in the same webContents force the OS
 * browser to an attacker URL — exactly the CVE-2026-70608 posture the handler
 * exists to hold.
 *
 * So the opener is identified for us, by construction: main injects this shim
 * into a frame it has ALREADY identified by process/routing id AND verified as
 * an exact hub origin on a committed picker navigation (see
 * `isBridgedPickerNavigation`). The shim posts the requested URL to the window,
 * and the renderer acts on it only when `event.origin` — set by Chromium, not
 * by the message — is the hub origin. An artifact frame has an opaque origin,
 * so it can never satisfy that check, and no window is ever created from a
 * guest on either path.
 */

/** The message type the injected shim posts to the renderer. */
export const HUB_PICKER_OPEN_MESSAGE = 'hermes-hub-open'

/**
 * Injected into a verified picker frame's main world on every committed
 * navigation onto a picker URL (so a reload or a re-route re-installs it).
 *
 * `window.open` for an http/https/mailto target — absolute or relative to the
 * frame — is posted instead of attempted; anything else falls through to the
 * native implementation, which the CVE handler then denies as before. Anchors
 * with `target="_blank"` are caught in the capture phase for the same reason
 * (they never call `window.open`). Returns a short string so the injection can
 * be asserted from a test harness.
 */
export const HUB_PICKER_OPEN_BRIDGE_SOURCE = `(() => {
  if (window.__hermesHubPickerBridge) { return 'present' }
  window.__hermesHubPickerBridge = true
  var SAFE = /^(https?:|mailto:)/i
  function resolve(url) {
    var target = url == null ? '' : String(url)
    if (!target) { return null }
    try { target = new URL(target, window.location.href).href } catch (e) { return null }
    return SAFE.test(target) ? target : null
  }
  function post(url) {
    try { window.parent.postMessage({ type: '${HUB_PICKER_OPEN_MESSAGE}', url: String(url) }, '*') } catch (e) {}
  }
  var nativeOpen = window.open
  window.open = function (url) {
    if (url == null || String(url) === '') { return null }
    var target = resolve(url)
    if (target) { post(target); return null }
    try { return nativeOpen.apply(window, arguments) } catch (e) { return null }
  }
  document.addEventListener('click', function (event) {
    var el = event.target
    var anchor = el && el.closest ? el.closest('a[href]') : null
    if (!anchor || String(anchor.target || '').toLowerCase() !== '_blank') { return }
    var target = resolve(anchor.href)
    if (!target) { return }
    event.preventDefault()
    event.stopPropagation()
    post(target)
  }, true)
  return 'installed'
})()`
