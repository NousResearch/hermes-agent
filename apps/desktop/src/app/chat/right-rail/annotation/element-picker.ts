/**
 * Element picker probe for the preview annotation layer.
 *
 * `PICKER_PROBE_SOURCE` is injected into the preview <webview> via
 * `webview.executeJavaScript()`. It must be fully self-contained: no imports,
 * no closures over module state, no TypeScript syntax.
 *
 * The probe:
 *  - paints a highlight overlay that follows the hovered element
 *  - dims the rest of the page with a shadow mask
 *  - on click, resolves the element under the cursor and returns its
 *    descriptor JSON via the executeJavaScript promise
 *  - on Escape, tears itself down and returns null
 *  - ignores clicks inside cross-origin iframes (reported as iframeBlocked)
 *
 * Communication protocol (single shot, not persistent): the host awaits the
 * returned Promise; the probe resolves it on pick / cancel. Navigation or a
 * second injection invalidates older probes automatically via the global key.
 */

export interface PickedElement {
  blockedReason?: 'cross-origin-iframe'
  classes: string[]
  id: string
  pageTitle: string
  pageUrl: string
  rect: { height: number; width: number; x: number; y: number }
  scrollX: number
  scrollY: number
  selector: string
  tagName: string
  text: string
}

export type PickerResult =
  | { cancelled: true; kind: 'cancelled' }
  | { kind: 'picked'; element: PickedElement }
  | { kind: 'region'; region: PickedRegion }
  | { kind: 'iframe-blocked' }

export interface PickedRegion {
  pageTitle: string
  pageUrl: string
  rect: { height: number; width: number; x: number; y: number }
  scrollX: number
  scrollY: number
}

export const PICKER_GLOBAL_KEY = '__hermesAnnotationPicker__'
export const OVERLAY_ID = '__hermes_annotation_overlay__'
export const BANNER_ID = '__hermes_annotation_banner__'

/**
 * Builds the probe source with a localized banner message baked in.
 * The probe runs in the target page, far from our i18n runtime, so the
 * message string is interpolated before injection.
 */
export function buildPickerProbeSource(bannerMessage: string): string {
  return PICKER_PROBE_SOURCE.replace('__HERMES_BANNER_MESSAGE__', () =>
    JSON.stringify(bannerMessage).slice(1, -1)
  )
}

export const PICKER_PROBE_SOURCE = `(function () {
  var GLOBAL_KEY = ${JSON.stringify(PICKER_GLOBAL_KEY)};
  var OVERLAY_ID = ${JSON.stringify(OVERLAY_ID)};
  var BANNER_ID = ${JSON.stringify(BANNER_ID)};

  // Only one live probe at a time — tear down any previous instance.
  if (window[GLOBAL_KEY] && typeof window[GLOBAL_KEY].teardown === 'function') {
    window[GLOBAL_KEY].teardown();
  }

  function escapeIdent(value) {
    if (window.CSS && typeof window.CSS.escape === 'function') {
      return window.CSS.escape(value);
    }
    return String(value).replace(/([^a-zA-Z0-9_-])/g, '\\\\$1');
  }

  var PREFIXED_HASH_RE = /^(?:css-[a-z0-9]+|sc-[a-z0-9]+)$/i;
  var BARE_HASH_RE = /^[a-z0-9]{6,}$/;

  function isStableClass(name) {
    if (!name || name.length < 2) return false;
    if (PREFIXED_HASH_RE.test(name)) return false;
    if (BARE_HASH_RE.test(name) && /\\d/.test(name)) return false;
    return true;
  }

  function stableClassesOf(el) {
    var out = [];
    for (var i = 0; i < el.classList.length && out.length < 3; i++) {
      if (isStableClass(el.classList[i])) out.push(el.classList[i]);
    }
    return out;
  }

  function stepFor(el) {
    if (el.id) return '#' + escapeIdent(el.id);
    var tag = el.tagName.toLowerCase();
    var classes = stableClassesOf(el);
    var parent = el.parentElement;
    var base = classes.length ? tag + '.' + classes.map(escapeIdent).join('.') : tag;
    if (parent) {
      var sameTag = [];
      for (var i = 0; i < parent.children.length; i++) {
        if (parent.children[i].tagName === el.tagName) sameTag.push(parent.children[i]);
      }
      if (sameTag.length > 1) {
        return base + ':nth-of-type(' + (sameTag.indexOf(el) + 1) + ')';
      }
    }
    return base;
  }

  function buildSelector(el) {
    if (el.id) return '#' + escapeIdent(el.id);
    var steps = [];
    var current = el;
    var depth = 0;
    while (current && current.tagName.toLowerCase() !== 'html' && depth < 8) {
      steps.unshift(stepFor(current));
      if (current.id) break;
      current = current.parentElement;
      depth++;
    }
    return steps.join(' > ');
  }

  // ---- overlay ----------------------------------------------------------
  var highlight = document.createElement('div');
  highlight.id = OVERLAY_ID;
  highlight.style.cssText = [
    'position:fixed',
    'pointer-events:none',
    'z-index:2147483647',
    'border:2px solid #ef4444',
    'background:rgba(239,68,68,0.12)',
    'box-shadow:0 0 0 4000px rgba(0,0,0,0.35)',
    'border-radius:2px',
    'transition:top 60ms ease,left 60ms ease,width 60ms ease,height 60ms ease',
    'display:none'
  ].join(';');

  var banner = document.createElement('div');
  banner.id = BANNER_ID;
  banner.style.cssText = [
    'position:fixed',
    'top:0',
    'left:0',
    'right:0',
    'z-index:2147483647',
    'pointer-events:none',
    'text-align:center',
    'padding:6px 0',
    'font:12px/1.4 -apple-system,Segoe UI,sans-serif',
    'color:#fff',
    'background:rgba(239,68,68,0.92)'
  ].join(';');
  banner.textContent = '\\u{1F4CC} __HERMES_BANNER_MESSAGE__';

  document.documentElement.appendChild(highlight);
  document.documentElement.appendChild(banner);

  var hovered = null;

  function paintHighlight(el) {
    var r = el.getBoundingClientRect();
    highlight.style.display = 'block';
    highlight.style.top = r.top + 'px';
    highlight.style.left = r.left + 'px';
    highlight.style.width = r.width + 'px';
    highlight.style.height = r.height + 'px';
  }

  function describe(el) {
    var r = el.getBoundingClientRect();
    var text = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 50);
    return {
      selector: buildSelector(el),
      tagName: el.tagName,
      id: el.id || '',
      classes: stableClassesOf(el),
      text: text,
      rect: { x: r.x, y: r.y, width: r.width, height: r.height },
      scrollX: window.scrollX,
      scrollY: window.scrollY,
      pageUrl: location.href,
      pageTitle: document.title
    };
  }

  function onMove(event) {
    if (typeof document.elementFromPoint !== 'function') return;
    var el = document.elementFromPoint(event.clientX, event.clientY);
    if (!el || el === highlight || el === banner) return;
    hovered = el;
    paintHighlight(el);
  }

  function onKey(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      teardown();
      resolve({ kind: 'cancelled', cancelled: true });
    }
  }

  var resolve;
  var promise = new Promise(function (res) { resolve = res; });

  // ---- unified listener registry ---------------------------------------
  var listeners = [];
  function listen(type, fn) {
    document.addEventListener(type, fn, true);
    listeners.push([type, fn]);
  }

  // ---- region drag selection -------------------------------------------
  var dragStart = null;
  var suppressClick = false;
  var DRAG_THRESHOLD_PX = 6;

  var marquee = document.createElement('div');
  marquee.style.cssText = [
    'position:fixed',
    'pointer-events:none',
    'z-index:2147483647',
    'border:2px dashed #ef4444',
    'background:rgba(239,68,68,0.10)',
    'display:none'
  ].join(';');
  document.documentElement.appendChild(marquee);

  function onMouseDown(event) {
    if (event.button !== 0) return;
    dragStart = { x: event.clientX, y: event.clientY };
  }

  function onDragMove(event) {
    if (!dragStart) return;
    var dx = event.clientX - dragStart.x;
    var dy = event.clientY - dragStart.y;
    if (Math.abs(dx) < DRAG_THRESHOLD_PX && Math.abs(dy) < DRAG_THRESHOLD_PX) return;

    highlight.style.display = 'none';
    marquee.style.display = 'block';
    marquee.style.left = Math.min(dragStart.x, event.clientX) + 'px';
    marquee.style.top = Math.min(dragStart.y, event.clientY) + 'px';
    marquee.style.width = Math.abs(dx) + 'px';
    marquee.style.height = Math.abs(dy) + 'px';
  }

  function onMouseUp(event) {
    if (!dragStart) return;

    var dx = event.clientX - dragStart.x;
    var dy = event.clientY - dragStart.y;
    var wasDrag = Math.abs(dx) >= DRAG_THRESHOLD_PX || Math.abs(dy) >= DRAG_THRESHOLD_PX;
    var start = dragStart;
    dragStart = null;
    marquee.style.display = 'none';

    if (wasDrag) {
      suppressClick = true;
      event.preventDefault();
      event.stopPropagation();
      var rect = {
        x: Math.min(start.x, event.clientX),
        y: Math.min(start.y, event.clientY),
        width: Math.abs(dx),
        height: Math.abs(dy)
      };
      teardown();
      resolve({
        kind: 'region',
        region: {
          rect: rect,
          scrollX: window.scrollX,
          scrollY: window.scrollY,
          pageUrl: location.href,
          pageTitle: document.title
        }
      });
    }
    // not a drag → let the click handler do its job
  }

  // Click handling: resolves with a single-element descriptor. The
  // suppressClick flag swallows the click that follows a completed drag.
  function onClick(event) {
    if (suppressClick) {
      event.preventDefault();
      event.stopPropagation();
      suppressClick = false;
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    var el = typeof document.elementFromPoint === 'function'
      ? document.elementFromPoint(event.clientX, event.clientY)
      : null;

    // Cross-origin iframe contents are unreachable — report and tear down.
    if (el && el.tagName === 'IFRAME') {
      try {
        var doc = el.contentDocument;
        if (!doc) {
          teardown();
          resolve({ kind: 'iframe-blocked' });
          return;
        }
      } catch (e) {
        teardown();
        resolve({ kind: 'iframe-blocked' });
        return;
      }
    }

    if (!el || el === highlight || el === banner || el === marquee) return;

    var descriptor = describe(el);
    teardown();
    resolve({ kind: 'picked', element: descriptor });
  }

  listen('mousemove', onMove);
  listen('mousemove', onDragMove);
  listen('mousedown', onMouseDown);
  listen('mouseup', onMouseUp);
  listen('click', onClick);
  listen('keydown', onKey);

  function teardown() {
    for (var i = 0; i < listeners.length; i++) {
      document.removeEventListener(listeners[i][0], listeners[i][1], true);
    }
    listeners = [];
    [highlight, banner, marquee].forEach(function (node) {
      if (node.parentNode) node.parentNode.removeChild(node);
    });
    try { delete window[GLOBAL_KEY]; } catch (e) { window[GLOBAL_KEY] = undefined; }
  }

  window[GLOBAL_KEY] = { teardown: teardown };

  return promise;
})()`
