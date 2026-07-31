/**
 * Persistent annotation-session probe for the preview webview.
 *
 * Unlike the single-shot picker (element-picker.ts), this probe stays
 * resident in the page for the whole annotation session: every pick / region
 * / badge interaction is reported to the host over the webview
 * `console-message` channel, and the host drives the probe (badges, picking
 * pause/resume, teardown) by calling `window.__hermesAnnotationSession__`
 * methods via `executeJavaScript`.
 *
 * Protocol (page → host): `console.log('__HERMES_ANNOTATE__:' + JSON.stringify(payload))`
 * where payload is a `SessionEvent`. The host filters console messages by the
 * channel prefix; everything else is left alone.
 *
 * The probe source must be fully self-contained: no imports, no closures over
 * module state, no TypeScript syntax.
 */

import type { PickedElement, PickedRegion } from './element-picker'

export type SessionEvent =
  | { clickX: number; clickY: number; kind: 'element'; target: PickedElement; type: 'pick' }
  | { clickX: number; clickY: number; kind: 'region'; target: PickedRegion; type: 'pick' }
  | { type: 'badge-click'; id: string }
  | { type: 'cancel-request' }
  | { type: 'iframe-blocked' }

export const SESSION_CHANNEL = '__HERMES_ANNOTATE__:'
export const SESSION_GLOBAL_KEY = '__hermesAnnotationSession__'

function isValidRect(rect: unknown): rect is { x: number; y: number; width: number; height: number } {
  if (rect == null || typeof rect !== 'object') {
    return false
  }

  const r = rect as Record<string, unknown>

  return typeof r.x === 'number' && typeof r.y === 'number' && typeof r.width === 'number' && typeof r.height === 'number'
}

export function parseSessionEvent(message: string): SessionEvent | null {
  if (!message.startsWith(SESSION_CHANNEL)) {
    return null
  }

  let raw: Record<string, unknown>

  try {
    raw = JSON.parse(message.slice(SESSION_CHANNEL.length))
  } catch {
    return null
  }

  if (raw == null || typeof raw !== 'object') {
    return null
  }

  // Discriminated-event validation — the page controls console output, so
  // every field that the host dereferences without a guard must be checked.
  if (raw.type === 'pick') {
    const kind = raw.kind

    if (kind !== 'element' && kind !== 'region') {
      return null
    }

    if (typeof raw.clickX !== 'number' || typeof raw.clickY !== 'number') {
      return null
    }

    const target = raw.target as Record<string, unknown> | null

    if (target == null || typeof target !== 'object' || !isValidRect(target.rect)) {
      return null
    }

    return {
      type: 'pick',
      kind,
      clickX: raw.clickX as number,
      clickY: raw.clickY as number,
      target
    } as unknown as SessionEvent
  }

  if (raw.type === 'badge-click') {
    if (typeof raw.id !== 'string') {
      return null
    }

    return { type: 'badge-click', id: raw.id }
  }

  if (raw.type === 'cancel-request') {
    return { type: 'cancel-request' }
  }

  if (raw.type === 'iframe-blocked') {
    return { type: 'iframe-blocked' }
  }

  return null
}

/** Build the host-side call that toggles picking while the comment card is open. */
export function buildSetPickingCall(active: boolean): string {
  return `window[${JSON.stringify(SESSION_GLOBAL_KEY)}] && window[${JSON.stringify(SESSION_GLOBAL_KEY)}].setPicking(${active ? 'true' : 'false'})`
}

export function buildAddBadgeCall(id: string, number: number, x: number, y: number): string {
  return `window[${JSON.stringify(SESSION_GLOBAL_KEY)}] && window[${JSON.stringify(SESSION_GLOBAL_KEY)}].addBadge(${JSON.stringify(id)}, ${number}, ${Math.round(x)}, ${Math.round(y)})`
}

export function buildRemoveBadgeCall(id: string): string {
  return `window[${JSON.stringify(SESSION_GLOBAL_KEY)}] && window[${JSON.stringify(SESSION_GLOBAL_KEY)}].removeBadge(${JSON.stringify(id)})`
}

export function buildTeardownCall(): string {
  return `window[${JSON.stringify(SESSION_GLOBAL_KEY)}] && window[${JSON.stringify(SESSION_GLOBAL_KEY)}].teardown()`
}

/** Flash a temporary highlight on a target rect and scroll it into view. */
export function buildFlashCall(rect: { height: number; width: number; x: number; y: number }): string {
  return `window[${JSON.stringify(SESSION_GLOBAL_KEY)}] && window[${JSON.stringify(SESSION_GLOBAL_KEY)}].flash(${Math.round(rect.x)}, ${Math.round(rect.y)}, ${Math.round(rect.width)}, ${Math.round(rect.height)})`
}

export function buildSessionProbeSource(bannerMessage: string): string {
  return SESSION_PROBE_SOURCE.replace('__HERMES_BANNER_MESSAGE__', () =>
    JSON.stringify(bannerMessage).slice(1, -1)
  )
}

export const SESSION_PROBE_SOURCE = `(function () {
  var GLOBAL_KEY = ${JSON.stringify(SESSION_GLOBAL_KEY)};
  var CHANNEL = ${JSON.stringify(SESSION_CHANNEL)};

  if (window[GLOBAL_KEY] && typeof window[GLOBAL_KEY].teardown === 'function') {
    window[GLOBAL_KEY].teardown();
  }

  function emit(payload) {
    console.log(CHANNEL + JSON.stringify(payload));
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

  // ---- overlays ---------------------------------------------------------
  var highlight = document.createElement('div');
  highlight.style.cssText = [
    'position:fixed',
    'pointer-events:none',
    'z-index:2147483646',
    'border:2px solid #ef4444',
    'background:rgba(239,68,68,0.12)',
    'box-shadow:0 0 0 4000px rgba(0,0,0,0.35)',
    'border-radius:2px',
    'transition:top 60ms ease,left 60ms ease,width 60ms ease,height 60ms ease',
    'display:none'
  ].join(';');

  var banner = document.createElement('div');
  banner.style.cssText = [
    'position:fixed',
    'top:0',
    'left:0',
    'right:0',
    'z-index:2147483646',
    'pointer-events:none',
    'text-align:center',
    'padding:6px 0',
    'font:12px/1.4 -apple-system,Segoe UI,sans-serif',
    'color:#fff',
    'background:rgba(239,68,68,0.92)'
  ].join(';');
  banner.textContent = '\\u{1F4CC} __HERMES_BANNER_MESSAGE__';

  var marquee = document.createElement('div');
  marquee.style.cssText = [
    'position:fixed',
    'pointer-events:none',
    'z-index:2147483646',
    'border:2px dashed #ef4444',
    'background:rgba(239,68,68,0.10)',
    'display:none'
  ].join(';');

  var badgeLayer = document.createElement('div');
  badgeLayer.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483645';

  document.documentElement.appendChild(highlight);
  document.documentElement.appendChild(banner);
  document.documentElement.appendChild(marquee);
  document.documentElement.appendChild(badgeLayer);

  // ---- state ------------------------------------------------------------
  var pickingActive = true;
  var badges = {};

  function paintHighlight(el) {
    var r = el.getBoundingClientRect();
    highlight.style.display = 'block';
    highlight.style.top = r.top + 'px';
    highlight.style.left = r.left + 'px';
    highlight.style.width = r.width + 'px';
    highlight.style.height = r.height + 'px';
  }

  function hideHighlight() {
    highlight.style.display = 'none';
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

  function isOwnNode(el) {
    return el === highlight || el === banner || el === marquee ||
      el === badgeLayer || (el && el.dataset && el.dataset.hermesBadge);
  }

  function onMove(event) {
    if (!pickingActive) return;
    if (typeof document.elementFromPoint !== 'function') return;
    var el = document.elementFromPoint(event.clientX, event.clientY);
    if (!el || isOwnNode(el)) return;
    paintHighlight(el);
  }

  function onKey(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      emit({ type: 'cancel-request' });
    }
  }

  // ---- region drag -------------------------------------------------------
  var dragStart = null;
  var suppressClick = false;
  var DRAG_THRESHOLD_PX = 6;

  function onMouseDown(event) {
    if (!pickingActive || event.button !== 0) return;
    dragStart = { x: event.clientX, y: event.clientY };
  }

  function onDragMove(event) {
    if (!dragStart) return;
    var dx = event.clientX - dragStart.x;
    var dy = event.clientY - dragStart.y;
    if (Math.abs(dx) < DRAG_THRESHOLD_PX && Math.abs(dy) < DRAG_THRESHOLD_PX) return;

    hideHighlight();
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
      // Freeze picking so the host screenshots a clean page before the
      // comment card opens.
      pickingActive = false;
      hideHighlight();
      emit({
        type: 'pick',
        kind: 'region',
        clickX: rect.x,
        clickY: rect.y,
        target: {
          rect: rect,
          scrollX: window.scrollX,
          scrollY: window.scrollY,
          pageUrl: location.href,
          pageTitle: document.title
        }
      });
    }
  }

  function onClick(event) {
    if (suppressClick) {
      event.preventDefault();
      event.stopPropagation();
      suppressClick = false;
      return;
    }
    if (!pickingActive) return;

    var el = typeof document.elementFromPoint === 'function'
      ? document.elementFromPoint(event.clientX, event.clientY)
      : null;

    if (el && el.dataset && el.dataset.hermesBadge) {
      event.preventDefault();
      event.stopPropagation();
      emit({ type: 'badge-click', id: el.dataset.hermesBadge });
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    if (el && el.tagName === 'IFRAME') {
      var blocked = false;
      try {
        if (!el.contentDocument) blocked = true;
      } catch (e) {
        blocked = true;
      }
      if (blocked) {
        emit({ type: 'iframe-blocked' });
        return;
      }
    }

    if (!el || isOwnNode(el)) return;

    var descriptor = describe(el);
    pickingActive = false;
    hideHighlight();
    emit({ type: 'pick', kind: 'element', clickX: event.clientX, clickY: event.clientY, target: descriptor });
  }

  // ---- badges ------------------------------------------------------------
  // The host owns all annotation state; the probe is a dumb renderer. Badge
  // coordinates are passed explicitly so the host can renumber/re-pin freely.
  function addBadge(id, number, x, y) {
    if (badges[id]) return;

    var badge = document.createElement('div');
    badge.dataset.hermesBadge = id;
    badge.textContent = String(number);
    badge.style.cssText = [
      'position:fixed',
      'pointer-events:auto',
      'cursor:pointer',
      'z-index:2147483645',
      'left:' + Math.max(4, x - 10) + 'px',
      'top:' + Math.max(4, y - 10) + 'px',
      'min-width:20px',
      'height:20px',
      'padding:0 5px',
      'border-radius:10px',
      'background:#ef4444',
      'color:#fff',
      'font:600 11px/20px -apple-system,Segoe UI,sans-serif',
      'text-align:center',
      'box-shadow:0 1px 4px rgba(0,0,0,0.35)'
    ].join(';');
    badgeLayer.appendChild(badge);
    badges[id] = badge;
  }

  function removeBadge(id) {
    var badge = badges[id];
    if (badge && badge.parentNode) {
      badge.parentNode.removeChild(badge);
    }
    delete badges[id];
  }

  // ---- unified listener registry ----------------------------------------
  var listeners = [];
  function listen(type, fn) {
    document.addEventListener(type, fn, true);
    listeners.push([type, fn]);
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
    [highlight, banner, marquee, badgeLayer].forEach(function (node) {
      if (node.parentNode) node.parentNode.removeChild(node);
    });
    badges = {};
    try { delete window[GLOBAL_KEY]; } catch (e) { window[GLOBAL_KEY] = undefined; }
  }

  var flashTimer = null;
  function flash(x, y, w, h) {
    // Rect is viewport-relative — pin it to absolute page coordinates first,
    // because the scroll below changes the viewport origin.
    var absX = x + window.scrollX;
    var absY = y + window.scrollY;
    var targetTop = Math.max(0, absY - (window.innerHeight - h) / 2);
    window.scrollTo({ top: targetTop, behavior: 'smooth' });

    // Paint the flash after the scroll settles, so coordinates line up.
    if (flashTimer) clearTimeout(flashTimer);
    flashTimer = setTimeout(function () {
      highlight.style.display = 'block';
      highlight.style.top = (absY - window.scrollY) + 'px';
      highlight.style.left = (absX - window.scrollX) + 'px';
      highlight.style.width = w + 'px';
      highlight.style.height = h + 'px';
      // Blink: toggle visibility a few times then hide.
      var blinks = 0;
      var blinkTimer = setInterval(function () {
        highlight.style.opacity = highlight.style.opacity === '0.2' ? '1' : '0.2';
        blinks++;
        if (blinks >= 5) {
          clearInterval(blinkTimer);
          highlight.style.display = 'none';
          highlight.style.opacity = '1';
        }
      }, 220);
    }, 350);
  }

  window[GLOBAL_KEY] = {
    teardown: teardown,
    addBadge: addBadge,
    removeBadge: removeBadge,
    flash: flash,
    setPicking: function (on) {
      pickingActive = !!on;
      if (!on) hideHighlight();
    }
  };

  return true;
})()`
