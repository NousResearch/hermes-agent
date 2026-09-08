/**
 * Read-only tool implementations for the Desktop Debug MCP server.
 *
 * Extracted from server.mjs following the module-per-concern layout used by
 * tools/act.mjs / tools/flows.mjs: server.mjs stays a thin MCP-wiring entry
 * point. All CDP interaction is injected via `deps.connect`, output bounds via
 * MAX_TEXT / MAX_NODES / MAX_EVAL.
 */

import { isConnectablePage, matchesTarget } from '../cdp-client.mjs'

/** Resolve a selector: either a SELECTORS key or a raw CSS selector. */
export const resolveSelector = (sel, SELECTORS) => (SELECTORS && SELECTORS[sel]) || sel

/** Friendly guard: a selector must be a non-empty, non-whitespace string. */
export function requireSelector(sel) {
  if (typeof sel !== 'string' || !sel.trim()) {
    throw new Error('selector required: pass a SELECTORS key (composer, threadViewport, ...) or a CSS selector')
  }
  return sel
}

/**
 * Build the read-tool implementations.
 * @param {object} deps { connect, MAX_TEXT, MAX_NODES, MAX_EVAL, port, allowAct, SELECTORS }
 */
export function createReadTools(deps) {
  const {
    connect,
    MAX_TEXT = 80,
    MAX_NODES = 20,
    MAX_EVAL = 4000,
    SELECTORS = {}
  } = deps

  /** Evaluate with a bounded JSON result. Throws with a friendly message on failure. */
  async function evalBounded(expression) {
    const c = await connect()
    const out = await c.eval(expression)
    // A2: JSON.stringify(undefined) is undefined (not a string) — a renderer
    // expression returning undefined must not crash the tool call.
    const s = typeof out === 'string' ? out : JSON.stringify(out) ?? 'null'
    if (s.length <= MAX_EVAL) return s
    const cut = s.length - MAX_EVAL
    return s.slice(0, MAX_EVAL) + `…[+${cut} chars truncated]`
  }

  async function status() {
    let alive = false
    let targets = []

    try {
      // fetchImpl is injectable for tests; production uses global fetch.
      const doFetch = deps.fetchImpl || fetch
      const list = await (await doFetch(`http://127.0.0.1:${deps.port}/json/list`)).json()
      // Two predicates, both shared with the CDP client: connectable (page +
      // WS url) AND the --match URL filter — status must never report a
      // target the client would refuse to connect to (Bugbot P2).
      targets = list
        .filter((t) => isConnectablePage(t) && matchesTarget(t.url, deps.match))
        .map((t) => ({ url: String(t.url).slice(0, 120), title: String(t.title).slice(0, 60) }))
      alive = targets.length > 0
    } catch {
      alive = false
    }

    return {
      cdpAlive: alive,
      port: deps.port,
      mode: alive ? 'dev' : 'unavailable',
      allowAct: deps.allowAct,
      match: deps.match ?? null,
      selectors: Object.keys(SELECTORS),
      targets
    }
  }

  async function inspect({ selector } = {}) {
    const sel = resolveSelector(requireSelector(selector), SELECTORS)
    return evalBounded(`(() => {
    const el = document.querySelector(${JSON.stringify(sel)})
    if (!el) return null
    const cs = getComputedStyle(el)
    const box = el.getBoundingClientRect()
    const ownClasses = typeof el.className === 'string' ? el.className : ''
    // Walk up a few ancestors: inherited styles are the classic "why won't it apply" trap.
    const parents = []
    let n = el.parentElement
    while (n && parents.length < 5) { parents.push(n.className); n = n.parentElement }
    return JSON.stringify({
      tag: el.tagName.toLowerCase(),
      id: el.id || undefined,
      classes: ownClasses,
      box: { x: Math.round(box.x), y: Math.round(box.y), w: Math.round(box.width), h: Math.round(box.height) },
      visible: !!(box.width || box.height) && cs.display !== 'none' && cs.visibility !== 'hidden',
      computed: { display: cs.display, position: cs.position, fontSize: cs.fontSize, fontWeight: cs.fontWeight, color: cs.color, background: cs.backgroundColor },
      inheritedHint: ownClasses ? 'own classes present' : 'NO own class — value is inherited; fix the ancestor rule',
      ancestors: parents
    })
  })()`)
  }

  async function query({ selector, limit } = {}) {
    const sel = resolveSelector(requireSelector(selector), SELECTORS)
    // BUG #6 fix: `limit || MAX_NODES` turned an explicit 0 into the cap and a
    // negative into an unhelpful empty slice; coerce non-finite/non-positive to cap.
    const cap = Number.isFinite(limit) && limit > 0 ? Math.min(Math.floor(limit), MAX_NODES) : MAX_NODES
    return evalBounded(`(() => {
    const els = [...document.querySelectorAll(${JSON.stringify(sel)})].slice(0, ${cap})
    return els.map((el, i) => {
      const b = el.getBoundingClientRect()
      const txt = (el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, ${MAX_TEXT})
      return { i, text: txt, w: Math.round(b.width), h: Math.round(b.height), visible: b.width > 0 }
    })
  })()`)
  }

  async function consoleLog({ level, sinceMs } = {}) {
    const cutoff = sinceMs ? Date.now() - sinceMs : 0
    let rows = (deps.consoleRing || []).filter((r) => r.t >= cutoff)
    if (level) rows = rows.filter((r) => r.level === level)
    return rows.slice(-(deps.MAX_CONSOLE || 50))
  }

  async function screenshot() {
    const c = await connect()
    const shot = await c.send('Page.captureScreenshot', { format: 'png' })
    // Return the capture as MCP image content — never write to disk (BUG #1
    // class). If a saved copy is needed, the caller persists the bytes.
    return {
      content: [
        { type: 'image', data: shot.data, mimeType: 'image/png' },
        { type: 'text', text: JSON.stringify({ bytes: shot.data.length }) }
      ]
    }
  }

  return { evalBounded, status, inspect, query, consoleLog, screenshot, resolveSelector: (s) => resolveSelector(s, SELECTORS) }
}
