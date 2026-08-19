/**
 * PREVIEW ACT ENGINE — the one function that performs an interaction inside a
 * page, so the agent can drive the in-app browser instead of only reading it.
 *
 * `elements` hands back a numbered inventory of what can be interacted with
 * (`@e1`, `@e2`, … — the same ref shape the browser_* tools use, so a model
 * that knows one knows the other) and parks the matching nodes on a holder.
 * Every other verb resolves its target from that holder, falling back to a raw
 * CSS selector.
 *
 * Injected into the preview webview as source (`actInPage.toString()`), so it
 * MUST stay self-contained: no imports, no closure references, no renderer
 * globals — everything arrives as a parameter. The structural types below
 * erase at compile time, so the stringified function stays plain JS.
 */

/** One interactable node, as the agent sees it. */
export interface PreviewElement {
  /** Present and true only when the control is non-interactive right now. */
  disabled?: boolean
  /** Human-readable label (aria-label, text, placeholder, value …). */
  label: string
  /** Stable-for-this-snapshot handle: '@e1', '@e2', … */
  ref: string
  /** Explicit ARIA role, else the tag name. */
  role: string
  /** CSS selector that resolves back to this node, for re-finding it later. */
  selector: string
  /** Current value of a form control, truncated. */
  value?: string
}

/** A normalized action. `kind` is the verb; the rest is per-verb payload. */
export interface PreviewActAction {
  /** scroll distance in px. Defaults to ~90% of the viewport height. */
  amount?: number
  key?: string
  kind: 'click' | 'elements' | 'press' | 'scroll' | 'type'
  /** Cap on the returned inventory. */
  max?: number
  ref?: string
  selector?: string
  /** type: press Enter (and submit the owning form) after entering text. */
  submit?: boolean
  text?: string
  to?: 'bottom' | 'top'
}

export interface PreviewActResult {
  /** What the action landed on, for the agent's own log. */
  acted?: string
  elements?: PreviewElement[]
  error?: string
  note?: string
  success: boolean
  title?: string
  /** Live document URL after the action — a change means it navigated. */
  url?: string
}

/** Where the surface keeps the last snapshot between actions (a window global
 *  in the preview page), so '@e5' still means something on the next call. */
export interface PreviewActHolder {
  nodes?: Element[]
  /** URL the snapshot was taken on; a navigation invalidates every ref. */
  url?: string
}

const ACT_MAX_ELEMENTS = 120

/** Run one action against `doc`, resolving refs through `holder`. Self-contained. */
export function actInPage(doc: Document, holder: PreviewActHolder, action: PreviewActAction): PreviewActResult {
  const win = doc.defaultView
  const here = doc.location ? doc.location.href : ''

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

  /** Identity-first selector, positional fallback — the caller re-finds nodes
   *  with this once the refs have gone stale. */
  const selectorFor = (el: Element): string => {
    if (el.id) {
      return '#' + cssEscape(el.id)
    }

    const testId = el.getAttribute('data-testid')

    if (testId) {
      return '[data-testid="' + cssEscape(testId) + '"]'
    }

    const path: string[] = []
    let node: Element | null = el

    while (node && node !== doc.body && path.length < 8) {
      if (node.id) {
        path.unshift('#' + cssEscape(node.id))

        break
      }

      const parent: Element | null = node.parentElement
      const index = parent ? Array.prototype.indexOf.call(parent.children, node) : -1

      path.unshift(node.tagName.toLowerCase() + (index >= 0 ? ':nth-child(' + (index + 1) + ')' : ''))
      node = parent
    }

    return path.join(' > ')
  }

  const visible = (el: Element): boolean => {
    const style = win && win.getComputedStyle ? win.getComputedStyle(el) : null

    if (style && (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0')) {
      return false
    }

    const rect = el.getBoundingClientRect()

    // Off-screen is fine (we scroll to it); collapsed to nothing is not.
    return rect.width >= 1 && rect.height >= 1
  }

  const valueOf = (el: Element): string => {
    const control = el as HTMLInputElement

    if (typeof control.value === 'string' && control.value) {
      return clamp(control.value, 60)
    }

    if (control.checked !== undefined && (el.tagName === 'INPUT' || el.getAttribute('role') === 'checkbox')) {
      return control.checked ? 'checked' : 'unchecked'
    }

    return ''
  }

  const collect = (max: number): PreviewElement[] => {
    const nodes: Element[] = []
    const elements: PreviewElement[] = []

    const candidates = doc.querySelectorAll(
      'a[href], button, input:not([type="hidden"]), select, textarea, summary, label[for], ' +
        '[role="button"], [role="link"], [role="checkbox"], [role="radio"], [role="tab"], ' +
        '[role="menuitem"], [role="switch"], [role="option"], [role="combobox"], [role="searchbox"], ' +
        '[role="textbox"], [contenteditable=""], [contenteditable="true"], [onclick], ' +
        '[tabindex]:not([tabindex="-1"])'
    )

    for (const el of candidates) {
      if (elements.length >= max || nodes.indexOf(el) !== -1 || !visible(el)) {
        continue
      }

      const tag = el.tagName.toLowerCase()
      const role = el.getAttribute('role') || (tag === 'input' ? 'input:' + ((el as HTMLInputElement).type || 'text') : tag)
      const label = labelOf(el)
      const value = valueOf(el)

      // A control with neither a label nor a value is not addressable in prose
      // — the agent could not tell it apart from its unlabelled neighbours.
      if (!label && !value) {
        continue
      }

      const entry: PreviewElement = {
        label,
        ref: '@e' + (elements.length + 1),
        role,
        selector: selectorFor(el)
      }

      if ((el as HTMLInputElement).disabled) {
        entry.disabled = true
      }

      if (value) {
        entry.value = value
      }

      nodes.push(el)
      elements.push(entry)
    }

    holder.nodes = nodes
    holder.url = here

    return elements
  }

  /** Resolve the action's target: a ref from the last snapshot, else a selector. */
  const resolve = (): { el?: Element; error?: string } => {
    const ref = (action.ref || '').trim()

    if (ref) {
      if (holder.url !== here) {
        return { error: 'The page navigated since the last snapshot, so ' + ref + ' no longer points anywhere. Call elements again.' }
      }

      const index = Number(ref.replace(/^@e/, '')) - 1
      const el = holder.nodes && holder.nodes[index]

      if (!el) {
        return { error: 'Unknown element ' + ref + '. Call elements to get current refs.' }
      }

      if (!doc.contains(el)) {
        return { error: ref + ' has been removed from the page since the last snapshot. Call elements again.' }
      }

      return { el }
    }

    const selector = (action.selector || '').trim()

    if (!selector) {
      return { error: 'Pass a ref from elements, or a CSS selector.' }
    }

    let el: Element | null = null

    try {
      el = doc.querySelector(selector)
    } catch {
      return { error: 'Not a valid CSS selector: ' + selector }
    }

    return el ? { el } : { error: 'No element matches ' + selector + '.' }
  }

  const describe = (el: Element) => {
    const label = labelOf(el)

    return el.tagName.toLowerCase() + (label ? ' "' + label + '"' : '')
  }

  const answer = (result: PreviewActResult): PreviewActResult => ({
    ...result,
    title: doc.title || '',
    url: doc.location ? doc.location.href : ''
  })

  const fail = (error: string) => answer({ error, success: false })

  /** Center of `el` in viewport coords, for pointer events that read position. */
  const pointAt = (el: Element) => {
    const rect = el.getBoundingClientRect()

    return { clientX: Math.round(rect.left + rect.width / 2), clientY: Math.round(rect.top + rect.height / 2) }
  }

  if (action.kind === 'elements') {
    const elements = collect(Math.max(1, Math.min(action.max || ACT_MAX_ELEMENTS, ACT_MAX_ELEMENTS)))

    return answer({
      elements,
      note: elements.length ? undefined : 'No interactive elements found — the page may still be loading.',
      success: true
    })
  }

  if (action.kind === 'scroll') {
    const target = action.ref || action.selector ? resolve() : {}

    if (target.error) {
      return fail(target.error)
    }

    const scroller = (target.el as HTMLElement | undefined) || null
    const page = Math.round((win ? win.innerHeight : 800) * 0.9)
    const by = action.to === 'top' ? -1e7 : action.to === 'bottom' ? 1e7 : (action.amount ?? page)

    if (scroller) {
      scroller.scrollBy({ behavior: 'auto', top: by })
    } else if (win) {
      win.scrollBy({ behavior: 'auto', top: by })
    }

    return answer({ acted: scroller ? 'scrolled ' + describe(scroller) : 'scrolled the page', success: true })
  }

  const target = resolve()

  if (target.error || !target.el) {
    return fail(target.error || 'No target.')
  }

  const el = target.el as HTMLElement

  if ((el as HTMLInputElement).disabled) {
    return fail(describe(el) + ' is disabled.')
  }

  if (el.scrollIntoView) {
    el.scrollIntoView({ block: 'center', inline: 'nearest' })
  }

  if (action.kind === 'click') {
    const init = { bubbles: true, cancelable: true, ...pointAt(el) }

    // Frameworks bind to the pointer/mouse pair as often as to click itself, so
    // replay the whole sequence; el.click() then runs the native activation
    // (following a link, toggling a checkbox, submitting a form) that a bare
    // synthetic MouseEvent would leave to chance.
    if (typeof PointerEvent === 'function') {
      el.dispatchEvent(new PointerEvent('pointerdown', init))
      el.dispatchEvent(new PointerEvent('pointerup', init))
    }

    el.dispatchEvent(new MouseEvent('mousedown', init))
    el.dispatchEvent(new MouseEvent('mouseup', init))
    el.click()

    return answer({ acted: 'clicked ' + describe(el), success: true })
  }

  if (action.kind === 'type') {
    const text = action.text ?? ''

    const editable = el.isContentEditable || (el.getAttribute('contenteditable') ?? 'false') !== 'false'

    el.focus()

    if (editable) {
      el.textContent = text
    } else if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      // Assign through the prototype's setter: React (and anything else that
      // tracks the DOM value) shadows `value` with its own accessor and ignores
      // an input event whose value it thinks it already wrote, so a plain
      // `el.value = …` types into a field that snaps back on the next render.
      const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
      const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set

      if (setter) {
        setter.call(el, text)
      } else {
        ;(el as HTMLInputElement).value = text
      }
    } else if (el.tagName === 'SELECT') {
      return fail(describe(el) + ' is a dropdown — click it and click the option you want.')
    } else {
      return fail(describe(el) + ' is not a text field.')
    }

    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))

    if (action.submit) {
      const enter = { bubbles: true, cancelable: true, code: 'Enter', key: 'Enter' }

      el.dispatchEvent(new KeyboardEvent('keydown', enter))
      el.dispatchEvent(new KeyboardEvent('keyup', enter))

      const form = (el as HTMLInputElement).form

      if (form) {
        if (form.requestSubmit) {
          form.requestSubmit()
        } else {
          form.submit()
        }
      }
    }

    return answer({ acted: 'typed into ' + describe(el) + (action.submit ? ' and submitted' : ''), success: true })
  }

  if (action.kind === 'press') {
    const key = action.key || ''

    if (!key) {
      return fail('Pass the key to press, e.g. "Enter" or "Escape".')
    }

    const init = { bubbles: true, cancelable: true, code: key.length === 1 ? 'Key' + key.toUpperCase() : key, key }

    el.focus()
    el.dispatchEvent(new KeyboardEvent('keydown', init))
    el.dispatchEvent(new KeyboardEvent('keyup', init))

    return answer({ acted: 'pressed ' + key + ' on ' + describe(el), success: true })
  }

  return fail('Unknown action: ' + String(action.kind))
}
