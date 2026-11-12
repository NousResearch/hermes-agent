/**
 * click-session-helpers.mjs
 *
 * Shared logic for scripts/click-session.mjs — the interactive CDP helper that
 * clicks a sidebar session by partial title and reads the post-click state.
 *
 * The macOS check-suite flake (#97982): the old script did a bare
 * `scrollIntoView()` (smooth-scroll default on some surfaces; scroll can be
 * dropped entirely under load) followed by a fixed `setTimeout(3000)` before
 * reading state — a dropped scroll silently missed the click, and the fixed
 * sleep both raced fast loads and wasted time on slow ones. The helpers here
 * make the two steps deterministic:
 *
 *   - buildFindClickExpression: instant scroll (`behavior: 'auto'`,
 *     `block: 'center'`) so the element is where we clicked it.
 *   - pollUntil: bounded visibility/readiness polling instead of a fixed
 *     sleep — it returns as soon as the predicate expression is true.
 *   - evaluateJsonValue: unwraps the nested CDP envelope
 *     ({ result: { result: { value } } }) the old script read one level short,
 *     so the post-click state log printed `undefined`.
 *
 * Dependency-free (plain node) so it runs under the vitest electron project.
 */

export const CLICK_POLL_INTERVAL_MS = 100
export const CLICK_POLL_TIMEOUT_MS = 3000

/**
 * Runtime.evaluate expression: find the first button-ish element whose text
 * contains `titleMatch`, instantly scroll it into view (center of the
 * viewport, no smooth animation to drop), and click it. Returns JSON:
 * { found, tried? } when there is no match, { found: true, tag, text }
 * when the click was issued.
 */
export function buildFindClickExpression(titleMatch) {
  return `
    (() => {
      const titleMatch = ${JSON.stringify(titleMatch)}
      const all = document.querySelectorAll('button, a, div[role="button"]')
      const found = [...all].find(el => (el.textContent || '').includes(titleMatch))
      if (!found) return JSON.stringify({ found: false, tried: titleMatch })
      found.scrollIntoView({ behavior: 'auto', block: 'center' })
      found.click()
      return JSON.stringify({ found: true, tag: found.tagName, text: (found.textContent || '').slice(0, 80) })
    })()
  `
}

/** Runtime.evaluate expression answering `true` once the session view is live
 *  (the composer for the clicked session has mounted). */
export const POST_CLICK_READY_EXPRESSION =
  'JSON.stringify(!!document.querySelector(\'[data-slot="composer-rich-input"]\'))'

/**
 * Unwrap the nested CDP envelope. `send()` resolves with the whole WebSocket
 * message: `{ id, result: { result: { type, value } } }`. Returns the parsed
 * JSON of `message.result.result.value`, or null when any level is missing.
 */
export function evaluateJsonValue(message) {
  const raw = message?.result?.result?.value ?? message?.result?.value ?? null
  if (raw == null) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

/**
 * Poll `predicate(send)` (a Runtime.evaluate expression that JSON-encodes a
 * boolean) until it answers true or `timeoutMs` elapses. Resolves true/false —
 * never throws on timeout — so a slow session surfaces as `ready: false`
 * state instead of a hung script.
 */
export async function pollUntil(send, expression, {
  timeoutMs = CLICK_POLL_TIMEOUT_MS,
  intervalMs = CLICK_POLL_INTERVAL_MS
} = {}) {
  const deadline = Date.now() + timeoutMs

  for (;;) {
    const message = await send('Runtime.evaluate', { expression, returnByValue: true })
    const value = evaluateJsonValue(message)
    if (value === true) return true
    if (Date.now() + intervalMs >= deadline) return false
    await new Promise(resolve => setTimeout(resolve, intervalMs))
  }
}
