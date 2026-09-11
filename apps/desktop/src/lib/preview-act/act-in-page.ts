/**
 * PREVIEW ACT ENGINE — the one function that performs an interaction inside a
 * page, so the agent can drive the in-app browser instead of only reading it.
 *
 * `elements` hands back an inventory of what can be interacted with, each under
 * a durable handle that says what it is — `btn-sign-in`, `inp-email` — and
 * parks the matching nodes on a holder. Every other verb resolves its target
 * from that holder. Once the agent has a baseline for a page, later looks
 * answer with a delta instead of the whole inventory (see `survey`).
 *
 * SELF-CONTAINMENT, and why the helpers arrive as an argument.
 *
 * The core is injected into the guest page as source, where module scope does
 * not exist: a single free identifier is a ReferenceError on every call, which
 * Electron reports only as "Script failed to execute". That much has always
 * been true. The subtler half is that the renderer build MINIFIES, so an
 * imported binding named here would be renamed to something the injected
 * prelude never declared — and it would break in packaged builds only, staying
 * green in dev and in tests. So the core names nothing it does not receive:
 * `kit` carries the naming, visibility, and identity helpers, and
 * `actEngineSource` is what assembles them into one injectable string.
 *
 * `actInPage` below is the ordinary in-process entry point. It is NOT
 * stringified and is free to import.
 */

import { identityKit } from './identity'
import type { IdentityKit } from './identity'
import { namingKit } from './naming'
import type { NamingKit } from './naming'
import type {
  PreviewActAction,
  PreviewActBinding,
  PreviewActDelta,
  PreviewActHolder,
  PreviewActResult,
  PreviewElement,
  PreviewElementChange
} from './types'
import { visibilityKit } from './visibility'
import type { VisibilityKit } from './visibility'

export type {
  PreviewActAction,
  PreviewActBinding,
  PreviewActDelta,
  PreviewActHolder,
  PreviewActResult,
  PreviewElement,
  PreviewElementChange
} from './types'

/** Everything the core needs that it cannot define for itself. */
export interface ActKit {
  identity: IdentityKit
  naming: NamingKit
  visibility: VisibilityKit
}

/** Assemble the kits for one document. Used in-process; the guest page gets
 *  the same three factories through `actEngineSource`. */
export function buildActKit(doc: Document): ActKit {
  const naming = namingKit(doc)

  return { identity: identityKit(naming), naming, visibility: visibilityKit(doc, doc.defaultView) }
}

/** Run one action against `doc`, resolving refs through `holder`. */
export function actInPage(doc: Document, holder: PreviewActHolder, action: PreviewActAction): PreviewActResult {
  return actInPageCore(doc, holder, action, buildActKit(doc))
}

/** The injectable core. Everything it touches is a parameter — see the
 *  self-containment note above. */
export function actInPageCore(
  doc: Document,
  holder: PreviewActHolder,
  action: PreviewActAction,
  kit: ActKit
): PreviewActResult {
  const { anchorOf, labelOf, selectorFor, stableOf, valueOf } = kit.naming
  const { affinity, coin, shifted } = kit.identity
  const { onScreen, visible } = kit.visibility

  // Declared inside, not at module scope: this body is stringified and eval'd
  // in the guest page, where a module-level constant is simply not defined.
  const maxElements = 120
  // How many nodes the OVERLAY may draw, as opposed to how many the agent gets
  // told about. Far higher, because an extra mark costs one rect read where an
  // extra inventory row costs tokens on every single call.
  const maxMarks = 600
  // How alike a remembered element and a fresh one have to be before the
  // handle moves across. Below it we mint a new handle instead: a re-render
  // costing the agent a re-read is a cheap mistake, and a handle silently
  // pointing at the wrong button is not.
  const rebindBar = 0.6
  const win = doc.defaultView
  const here = doc.location ? doc.location.href : ''

  // Passing 'smooth' explicitly OVERRIDES the user's OS-level reduce-motion
  // setting, so ask before animating over someone who opted out.
  const still = !!(win && win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const glide: ScrollBehavior = still ? 'auto' : 'smooth'

  /** Walk the page and hand back what is interactable, in document order. The
   *  handles are assigned afterwards, by `survey`. */
  const sight = (max: number) => {
    const nodes: Element[] = []
    const field: Element[] = []
    const elements: PreviewElement[] = []

    const candidates = doc.querySelectorAll(
      'a[href], button, input:not([type="hidden"]), select, textarea, summary, label[for], ' +
        '[role="button"], [role="link"], [role="checkbox"], [role="radio"], [role="tab"], ' +
        '[role="menuitem"], [role="switch"], [role="option"], [role="combobox"], [role="searchbox"], ' +
        '[role="textbox"], [contenteditable=""], [contenteditable="true"], [onclick], ' +
        '[tabindex]:not([tabindex="-1"])'
    )

    for (const el of candidates) {
      if (elements.length >= max && field.length >= maxMarks) {
        break
      }

      if (!visible(el)) {
        continue
      }

      // Drawable from here on. The two lists diverge deliberately: the field is
      // what is on screen to be outlined, the inventory is what the agent can
      // name and reach — which includes things below the fold it will scroll to.
      if (field.length < maxMarks && onScreen(el)) {
        field.push(el)
      }

      if (elements.length >= max) {
        continue
      }

      const tag = el.tagName.toLowerCase()

      const role =
        el.getAttribute('role') || (tag === 'input' ? 'input:' + ((el as HTMLInputElement).type || 'text') : tag)

      const label = labelOf(el)
      const value = valueOf(el)

      // A control with neither a label nor a value is not addressable in prose
      // — the agent could not tell it apart from its unlabelled neighbours.
      // It is also the one case a durable handle cannot be minted for, since
      // there would be nothing to name it after and nothing to re-find it by,
      // so dropping it here keeps every handle we DO mint anchorable.
      if (!label && !value) {
        continue
      }

      const entry: PreviewElement = {
        label,
        // Filled in by `survey`, which is what knows whether this element
        // already has a handle.
        ref: '',
        role
      }

      const selector = selectorFor(el)

      if (selector) {
        entry.selector = selector
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
    holder.field = field

    return { elements, nodes }
  }

  /** Look at the page and say what is there — or, once there is something to
   *  compare against, only what moved.
   *
   *  Re-sending the whole inventory every action is what took a ten-step
   *  session from 45k to 85k tokens of context: the page barely changes between
   *  a scroll and a click, and the agent was being charged for a fresh copy of
   *  it each time. */
  const survey = (max: number): PreviewActResult => {
    // A navigation is a different page. Every handle on the old one is retired
    // rather than rebound onto whatever now sits in the same place.
    const fresh = holder.url !== here

    if (fresh) {
      holder.book = []
      holder.coined = {}
    }

    const book = holder.book || (holder.book = [])
    const coined = holder.coined || (holder.coined = {})
    const seen = sight(max)
    const claimed: Record<string, boolean> = {}
    const kept: PreviewActBinding[] = []
    const added: PreviewElement[] = []
    const changed: PreviewElementChange[] = []
    const rebound: string[] = []
    const known = new Map<Element, PreviewActBinding>()
    const waiting: number[] = []
    let same = 0

    for (const bound of book) {
      known.set(bound.el, bound)
    }

    // Pass one: the element object itself is still the one we remember. Free,
    // and it is what happens on a scroll, a hover, and most clicks.
    for (let i = 0; i < seen.elements.length; i++) {
      const entry = seen.elements[i]
      const bound = known.get(seen.nodes[i])

      if (!bound) {
        waiting.push(i)

        continue
      }

      const moved = shifted(bound, entry)

      entry.ref = bound.ref
      claimed[bound.ref] = true
      kept.push(bound)

      if (!moved) {
        same++

        continue
      }

      bound.label = entry.label
      bound.name = entry.label || entry.value || ''
      bound.off = !!entry.disabled
      bound.value = entry.value || ''
      changed.push(moved)
    }

    // Pass two: whatever is left either replaced something (a framework threw
    // the node away and built a new one) or is genuinely new. The pool is only
    // the handles whose element is GONE, which is both the correct candidate
    // set and a small one.
    const pool = book.filter(bound => !claimed[bound.ref] && !doc.contains(bound.el))

    for (const i of waiting) {
      const entry = seen.elements[i]
      const el = seen.nodes[i]

      const now: PreviewActBinding = {
        el,
        label: entry.label,
        name: entry.label || entry.value || '',
        off: !!entry.disabled,
        path: anchorOf(el),
        ref: '',
        role: entry.role,
        stable: stableOf(el),
        value: entry.value || ''
      }

      let best: PreviewActBinding | undefined
      let score = 0

      for (const bound of pool) {
        if (claimed[bound.ref]) {
          continue
        }

        const rung = affinity(bound, now)

        // Strictly better, so a tie goes to whichever candidate the page put
        // first and the same page twice re-binds the same way.
        if (rung >= rebindBar && rung > score) {
          best = bound
          score = rung
        }
      }

      if (best) {
        // Same handle, new node. Reported as one word rather than a removal
        // and an addition, because from the agent's side nothing happened —
        // its handle still works and it has nothing to re-read.
        entry.ref = best.ref
        best.el = el
        best.label = now.label
        best.name = now.name
        best.off = now.off
        best.path = now.path
        best.stable = now.stable
        best.value = now.value
        claimed[best.ref] = true
        kept.push(best)
        rebound.push(best.ref)

        continue
      }

      now.ref = coin(coined, entry.role, now.name)
      entry.ref = now.ref
      claimed[now.ref] = true
      kept.push(now)
      added.push(entry)
    }

    // A handle nobody claimed is gone ONLY if its element really left. One that
    // is still on the page but fell past `max` keeps working and is simply not
    // mentioned — saying "removed" about something the agent can still click
    // would be worse than saying nothing.
    const removed: string[] = []

    for (const bound of book) {
      if (claimed[bound.ref]) {
        continue
      }

      if (doc.contains(bound.el) && visible(bound.el)) {
        kept.push(bound)

        continue
      }

      removed.push(bound.ref)
    }

    holder.book = kept
    holder.url = here

    // The delta has to actually be cheaper. When half the page is new there is
    // nothing to reuse, and a delta is then just the inventory with extra
    // framing around it.
    const churn = added.length + changed.length

    if (fresh || action.full || churn * 2 >= seen.elements.length) {
      return { elements: seen.elements, success: true }
    }

    const delta: PreviewActDelta = { same }

    if (added.length) {
      delta.added = added
    }

    if (changed.length) {
      delta.changed = changed
    }

    if (removed.length) {
      delta.removed = removed
    }

    if (rebound.length) {
      delta.rebound = rebound
    }

    return { delta, success: true }
  }

  /** Resolve the action's target: a handle from the book, else a selector. */
  const resolve = (): { el?: Element; error?: string } => {
    const ref = (action.ref || '').trim()

    if (ref) {
      if (holder.url !== here) {
        return {
          error:
            'The page navigated since the last snapshot, so ' + ref + ' no longer points anywhere. Call elements again.'
        }
      }

      const bound = (holder.book || []).filter(entry => entry.ref === ref)[0]

      if (!bound) {
        return { error: 'Unknown element ' + ref + '. Call elements to get current refs.' }
      }

      if (!doc.contains(bound.el)) {
        return { error: ref + ' has been removed from the page since the last snapshot. Call elements again.' }
      }

      return { el: bound.el }
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
    const looked = survey(Math.max(1, Math.min(action.max || maxElements, maxElements)))
    const empty = !looked.delta && !(looked.elements || []).length

    return answer({
      ...looked,
      note: empty ? 'No interactive elements found — the page may still be loading.' : undefined
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
      scroller.scrollBy({ behavior: glide, top: by })
    } else if (win) {
      win.scrollBy({ behavior: glide, top: by })
    }

    return answer({ acted: scroller ? 'scrolled ' + describe(scroller) : 'scrolled the page', success: true })
  }

  const target = resolve()

  if (target.error || !target.el) {
    return fail(target.error || 'No target.')
  }

  const el = target.el as HTMLElement

  // Park the target where the watch overlay can find it, before anything can
  // fail — a ring around the thing that turned out to be disabled is exactly
  // the feedback someone watching wants.
  holder.aimed = el

  // 'locate' is the look-before-you-act half of an action: it brings the target
  // on screen and names it, so the overlay has something to draw while the real
  // verb is still a beat away.
  if (action.kind === 'locate') {
    if (el.scrollIntoView) {
      el.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' })
    }

    if (action.focus) {
      el.focus()
    }

    const spot = pointAt(el)
    const tag = el.tagName

    return answer({
      acted: 'looking at ' + describe(el),
      point: { x: spot.clientX, y: spot.clientY },
      success: true,
      // Real typing starts with a triple-click to clear the field. On anything
      // that is not a field that gesture selects the paragraph under it
      // instead, which is how the agent ended up highlighting whole pages.
      typable: tag === 'TEXTAREA' || tag === 'INPUT' || el.isContentEditable === true
    })
  }

  // Instant, unlike the `scroll` verb above. Getting a target on screen is
  // plumbing for the click that follows, not something anyone asked to watch —
  // and every millisecond of it is time the caller spends waiting for the page
  // to stop moving before it can aim real input at a fixed coordinate.
  if (el.scrollIntoView) {
    el.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' })
  }

  // Hovering a disabled control is allowed — a disabled button with a tooltip
  // explaining WHY is exactly the thing worth hovering.
  if (action.kind === 'hover') {
    const init = { bubbles: true, cancelable: true, ...pointAt(el) }

    if (typeof PointerEvent === 'function') {
      el.dispatchEvent(new PointerEvent('pointerover', init))
      el.dispatchEvent(new PointerEvent('pointermove', init))
    }

    el.dispatchEvent(new MouseEvent('mouseover', init))
    el.dispatchEvent(new MouseEvent('mousemove', init))

    return answer({ acted: 'hovered over ' + describe(el), success: true })
  }

  if ((el as HTMLInputElement).disabled) {
    return fail(describe(el) + ' is disabled.')
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

/** The engine as one injectable expression: `function (doc, holder, action)`.
 *
 *  The four factories are stringified together so the core's `kit` is built
 *  inside the guest page, out of sources that travelled with it. This is the
 *  ONLY supported way to get the engine's source — taking `.toString()` of any
 *  one piece yields a function whose helpers are not defined, which fails at
 *  call time rather than at injection. */
export function actEngineSource(): string {
  return `(function (doc, holder, action) {
  var naming = (${namingKit.toString()})(doc);
  var kit = {
    naming: naming,
    identity: (${identityKit.toString()})(naming),
    visibility: (${visibilityKit.toString()})(doc, doc.defaultView)
  };
  return (${actInPageCore.toString()})(doc, holder, action, kit);
})`
}
