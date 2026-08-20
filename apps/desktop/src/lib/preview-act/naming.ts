/**
 * NAMING — what an element is called, and what it is called by.
 *
 * Everything here turns an element into a short string: the accessible name,
 * the durable handle's stem and slug, the landmark it sits under, the identity
 * selector. No page state, no holder, no side effects.
 *
 * SELF-CONTAINMENT. `namingKit` is stringified into the guest page along with
 * the engine that uses it (see `act-in-page.ts`'s contract). Module scope does
 * not exist there, so the factory closes over nothing but its own argument and
 * declares every helper inside itself. A single free identifier is a
 * ReferenceError on every call, and Electron reports it only as "Script failed
 * to execute".
 *
 * That constraint is also why this is a factory rather than nine exports: nine
 * exports would be nine names for the bundler to mangle independently of the
 * call sites inside the stringified engine. One function that stringifies whole
 * has no cross-module references to break.
 */

export interface NamingKit {
  /** Where the element sits, coarsely — the nearest landmark. */
  anchorOf(el: Element): string
  clamp(text: string, max: number): string
  cssEscape(value: string): string
  /** The accessible name, by the usual ladder, truncated. */
  labelOf(el: Element): string
  /** An `#id` or `[data-testid]`, or empty when the page offers neither. */
  selectorFor(el: Element): string
  /** Lowercase, hyphenated, and short enough to read at a glance. */
  slug(name: string): string
  /** The strongest identity signal the page offers, if it offers one. */
  stableOf(el: Element): string
  /** The handle stem for a role: `btn`, `inp`, `lnk` … */
  stemOf(role: string): string
  /** A form control's current value, or its checked state. */
  valueOf(el: Element): string
}

/** Build the naming helpers against one document. */
export function namingKit(doc: Document): NamingKit {
  const cssEscape = (value: string) =>
    typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(value) : value.replace(/["\\]/g, '\\$&')

  const clamp = (text: string, max: number) => (text.length > max ? text.slice(0, max - 1) + '…' : text)

  const labelOf = (el: Element): string => {
    const aria = el.getAttribute('aria-label')

    if (aria) {
      return clamp(aria, 80)
    }

    const labelledBy = el.getAttribute('aria-labelledby')
    const labelled = labelledBy ? doc.getElementById(labelledBy) : null
    const text = ((labelled || el).textContent || '').trim().replace(/\s+/g, ' ')

    if (text) {
      return clamp(text, 80)
    }

    for (const attr of ['placeholder', 'title', 'alt', 'name', 'value']) {
      const value = el.getAttribute(attr)

      if (value) {
        return clamp(value, 80)
      }
    }

    return ''
  }

  /** The strongest identity signal the page offers, if it offers one. Nothing a
   *  re-render does to the surrounding markup disturbs these. */
  const stableOf = (el: Element): string =>
    el.id ||
    el.getAttribute('data-testid') ||
    el.getAttribute('name') ||
    el.getAttribute('aria-label') ||
    ''

  /** Handle stems by role, so a handle says what it is before it says which
   *  one. Anything unrecognised is `el`. */
  const stemOf = (role: string): string => {
    if (role === 'button' || role === 'summary') {
      return 'btn'
    }

    if (role === 'a' || role === 'link') {
      return 'lnk'
    }

    if (role === 'input:search' || role === 'searchbox') {
      return 'srch'
    }

    if (role === 'input:checkbox' || role === 'checkbox') {
      return 'chk'
    }

    if (role === 'input:radio' || role === 'radio') {
      return 'rdo'
    }

    if (role === 'select' || role === 'combobox') {
      return 'sel'
    }

    if (role === 'textarea') {
      return 'txt'
    }

    if (role === 'switch') {
      return 'sw'
    }

    if (role === 'tab' || role === 'menuitem' || role === 'option') {
      return role === 'menuitem' ? 'mi' : role === 'option' ? 'opt' : 'tab'
    }

    // Every `input:*` that isn't one of the special cases above, plus the ARIA
    // textbox. A date picker and an email field are both places text goes.
    return role.indexOf('input') === 0 || role === 'textbox' ? 'inp' : 'el'
  }

  /** Lowercase, hyphenated, and short enough to read at a glance. */
  const slug = (name: string): string => {
    let out = ''
    let dash = false

    for (let i = 0; i < name.length && out.length < 24; i++) {
      const ch = name[i]

      if (/[a-zA-Z0-9]/.test(ch)) {
        out += ch.toLowerCase()
        dash = false
      } else if (!dash && out) {
        out += '-'
        dash = true
      }
    }

    return out.replace(/-+$/, '')
  }

  /** Where the element sits, coarsely: the nearest landmark plus its position
   *  among same-role elements inside it. Deliberately NOT the CSS selector
   *  below — a wrapper div appearing anywhere in the chain changes that string
   *  completely, which is exactly the churn a re-bind has to see through. */
  const anchorOf = (el: Element): string => {
    const near = el.closest(
      'main,nav,header,footer,aside,[role="main"],[role="navigation"],[role="banner"],' +
        '[role="contentinfo"],[role="complementary"],[role="search"],form[aria-label],section[aria-label]'
    )

    if (!near) {
      return 'root'
    }

    const named = near.getAttribute('aria-label') || ''

    return near.tagName.toLowerCase() + (named ? '#' + slug(named) : '')
  }

  /** The element's own selector, when the page gives it one worth having.
   *
   *  Deliberately identity-only. This used to fall back to a chain of up to
   *  eight `:nth-child` rungs, and on a real app shell that column was 74% of
   *  the entire inventory — the single biggest thing the agent was paying for.
   *  It bought nothing: nothing downstream reads it, a positional chain is
   *  wrong the moment a sibling appears, and re-finding a node is what the
   *  durable ref now does properly. An `#id` is short, stable, and the one
   *  case where naming the node is genuinely useful to the model. */
  const selectorFor = (el: Element): string => {
    if (el.id) {
      return '#' + cssEscape(el.id)
    }

    const testId = el.getAttribute('data-testid')

    return testId ? '[data-testid="' + cssEscape(testId) + '"]' : ''
  }

  const valueOf = (el: Element): string => {
    const control = el as HTMLInputElement

    // Tickable controls answer with their state, and are asked FIRST: a
    // checkbox's `.value` is the string "on" unless the page sets one, so
    // reading the value first made a ticked box and an empty one identical to
    // the agent — the one thing about a checkbox worth reporting. Gated on the
    // input's TYPE, not on `checked` being defined, which it is (as false) on
    // every input including text fields.
    const kind = (control.type || '').toLowerCase()

    if (kind === 'checkbox' || kind === 'radio') {
      return control.checked ? 'checked' : 'unchecked'
    }

    // The same control built out of a div, where the state lives in ARIA.
    const role = el.getAttribute('role') || ''

    if (role === 'checkbox' || role === 'radio' || role === 'switch') {
      return el.getAttribute('aria-checked') === 'true' ? 'checked' : 'unchecked'
    }

    if (typeof control.value === 'string' && control.value) {
      return clamp(control.value, 60)
    }

    return ''
  }

  return { anchorOf, clamp, cssEscape, labelOf, selectorFor, slug, stableOf, stemOf, valueOf }
}
