/**
 * Rich-clipboard payload for an assistant reply.
 *
 * Inspired by Poke/Cognition's Sep 2026 change ("copying a message now preserves
 * formatting when pasted into email or documents, still pastes as Markdown into
 * code editors"): the clipboard carries BOTH `text/plain` (the raw Markdown the
 * agent wrote) and `text/html` (the prose as rendered). Rich targets (mail,
 * docs, chat apps) take the HTML; plain targets (editors, terminals) keep the
 * Markdown, so nothing changes for the workflow that copies replies into code.
 *
 * The HTML comes from the DOM the user is looking at rather than a second
 * Markdown pass, so tables, alerts, math and link resolution match the thread
 * exactly. Only prose blocks are taken: reasoning, tool cards and status rows
 * are chrome, not the reply.
 */

const MESSAGE_SCOPE_SELECTOR = '[data-slot="aui_response-group"], [data-slot="aui_assistant-message-root"]'
const PROSE_SELECTOR = '[data-slot="aui_assistant-message-content"] .aui-md.prose'

/** Prose that is not the reply itself: thinking, tool output, live status. */
const EXCLUDED_ANCESTOR_SELECTOR =
  '[data-slot="aui_reasoning-text"], [data-slot="aui_thinking-body"], [data-slot="aui_thinking-disclosure"], [data-slot="aui_turn-activity"], [data-slot="aui_msg-actions"]'

/** Interactive / decorative chrome that must not travel with the text. */
const CHROME_SELECTOR = 'button, [role="button"], svg, [aria-hidden="true"], [data-slot="code-card-icon"]'
/** Attributes that carry meaning for a pasted document; everything else (class,
 *  style, data-*, aria-*) is theme plumbing the target editor must not inherit. */
const KEPT_ATTRIBUTES = new Set(['alt', 'colspan', 'href', 'rowspan', 'src', 'start', 'title'])

function flattenCodeCards(root: HTMLElement): void {
  for (const card of Array.from(root.querySelectorAll('[data-slot="code-card"]'))) {
    const body = card.querySelector('[data-slot="code-card-body"]') ?? card
    const pre = root.ownerDocument.createElement('pre')
    const code = root.ownerDocument.createElement('code')

    // Shiki renders one span per token; textContent folds them back into the
    // source text (whitespace included, `pre` keeps it).
    code.textContent = body.textContent ?? ''
    pre.appendChild(code)
    card.replaceWith(pre)
  }
}

function stripPresentation(root: HTMLElement): void {
  for (const el of [root, ...Array.from(root.querySelectorAll('*'))]) {
    for (const attr of Array.from(el.attributes)) {
      if (!KEPT_ATTRIBUTES.has(attr.name)) {
        el.removeAttribute(attr.name)
      }
    }
  }
}

/**
 * HTML for the reply that owns `anchor` (any element inside the message, e.g.
 * the Copy button), or null when no rendered prose is reachable — callers then
 * copy Markdown only, exactly as before.
 */
export function renderedMessageHtml(anchor: Element | null | undefined): string | null {
  const scope = anchor?.closest(MESSAGE_SCOPE_SELECTOR)

  if (!scope) {
    return null
  }

  const blocks = Array.from(scope.querySelectorAll<HTMLElement>(PROSE_SELECTOR)).filter(
    block => !block.closest(EXCLUDED_ANCESTOR_SELECTOR)
  )

  if (blocks.length === 0) {
    return null
  }

  const html = blocks
    .map(block => {
      const clone = block.cloneNode(true) as HTMLElement

      for (const chrome of Array.from(clone.querySelectorAll(CHROME_SELECTOR))) {
        chrome.remove()
      }

      flattenCodeCards(clone)
      stripPresentation(clone)

      return clone.innerHTML.trim()
    })
    .filter(Boolean)
    .join('\n')

  return html || null
}
