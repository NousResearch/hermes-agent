const SESSION_SELECTOR = 'button, a, div[role="button"]'
const VISIBILITY_TIMEOUT_MS = 3_000
const VISIBILITY_POLL_MS = 50

export function readEvaluationValue(response) {
  return response?.result?.result?.value
}

/**
 * Build a browser expression that scrolls a matching session into view and
 * clicks it only after the renderer reports that it is visible.
 *
 * The expression is evaluated through the Chrome DevTools Protocol, so it
 * returns a promise rather than relying on Playwright-only helpers.
 */
export function buildClickSessionExpression(title) {
  const titleMatch = JSON.stringify(title)

  return `(async () => {
    const titleMatch = ${titleMatch}
    const all = document.querySelectorAll(${JSON.stringify(SESSION_SELECTOR)})
    const found = [...all].find(el => (el.textContent || '').includes(titleMatch))
    if (!found) return { found: false, tried: titleMatch }

    found.scrollIntoView({ behavior: 'auto', block: 'center' })

    const deadline = Date.now() + ${VISIBILITY_TIMEOUT_MS}
    while (Date.now() < deadline) {
      const rect = found.getBoundingClientRect()
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight
      const viewportWidth = window.innerWidth || document.documentElement.clientWidth
      const visible =
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= viewportHeight &&
        rect.right <= viewportWidth

      if (visible) {
        found.click()
        return {
          found: true,
          visible: true,
          tag: found.tagName,
          text: (found.textContent || '').slice(0, 80),
        }
      }

      await new Promise(resolve => setTimeout(resolve, ${VISIBILITY_POLL_MS}))
      found.scrollIntoView({ behavior: 'auto', block: 'center' })
    }

    return {
      found: true,
      visible: false,
      tag: found.tagName,
      text: (found.textContent || '').slice(0, 80),
    }
  })()`
}
