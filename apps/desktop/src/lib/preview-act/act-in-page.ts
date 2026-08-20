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
  /** Durable handle for as long as this page is open: 'btn-sign-in'. Legible on
   *  purpose — see the ref-minting note in `actInPage`. */
  ref: string
  /** Explicit ARIA role, else the tag name. */
  role: string
  /** The element's `#id` or `[data-testid]`, when it has one. Absent otherwise
   *  — address the element by its `ref`. */
  selector?: string
  /** Current value of a form control, truncated. */
  value?: string
}

/** An element that is still itself but no longer reads the same.
 *
 *  Only the fields that actually moved are present. Role and selector are
 *  absent by construction rather than by omission: a change in either would
 *  mean this is a different element, which the re-bind ladder would have
 *  refused to match in the first place. */
export interface PreviewElementChange {
  /** Present only when the control's availability flipped. */
  disabled?: boolean
  label?: string
  ref: string
  value?: string
}

/** What changed on the page since the last look. Sent instead of the whole
 *  inventory once the agent has a baseline for the page — see `survey`. */
export interface PreviewActDelta {
  /** Elements seen for the first time, in full. */
  added?: PreviewElement[]
  /** Same handle, new label/value/disabled state — and nothing else. */
  changed?: PreviewElementChange[]
  /** Handles that are gone from the page. */
  removed?: string[]
  /** Handles whose element was destroyed and recreated by a re-render. The
   *  handle still works; nothing about them needs re-reading. */
  rebound?: string[]
  /** How many handles were on the page and untouched. */
  same?: number
}

/** A normalized action. `kind` is the verb; the rest is per-verb payload. */
export interface PreviewActAction {
  /** scroll distance in px. Defaults to ~90% of the viewport height. */
  amount?: number
  key?: string
  /** `pin`/`unpin`/`hold` never reach the engine — they resolve their targets
   *  through `locate`/`elements` and then talk to the overlay — but they arrive
   *  on the same wire. */
  kind:
    | 'click'
    | 'elements'
    | 'hold'
    | 'hover'
    | 'locate'
    | 'pin'
    | 'press'
    | 'scroll'
    | 'strobe'
    | 'type'
    | 'unpin'
  /** locate: also give the target keyboard focus, for a key press that must not
   *  be preceded by a click (which would activate the control instead). */
  focus?: boolean
  /** elements: answer with the whole inventory rather than a delta. */
  full?: boolean
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
  /** What moved since the last look. Present INSTEAD of `elements` once the
   *  agent holds a baseline for this page. */
  delta?: PreviewActDelta
  /** The full inventory. Sent on the first look at a page, and again whenever
   *  the page changed too much for a delta to be the cheaper answer. */
  elements?: PreviewElement[]
  error?: string
  note?: string
  /** Viewport centre of a located target, for aiming real pointer input at it. */
  point?: { x: number; y: number }
  success: boolean
  title?: string
  /** locate: whether the target actually takes typed text. */
  typable?: boolean
  /** Live document URL after the action — a change means it navigated. */
  url?: string
}

/** One element the agent has a handle on, remembered across actions. */
export interface PreviewActBinding {
  el: Element
  /** What it read as last time. Kept field by field rather than as one hash so
   *  a change can be reported as only the part that moved. */
  label: string
  /** The accessible name at mint time, for re-finding this element after a
   *  re-render destroys and recreates its node. */
  name: string
  /** Whether the control was unavailable last time. */
  off: boolean
  /** Nearest-landmark path plus position among same-role siblings. */
  path: string
  ref: string
  role: string
  /** `id` / `name` / `data-testid` / `aria-label`, if the page provides one.
   *  The strongest re-bind signal there is, and the only one a rewrite of the
   *  surrounding markup cannot disturb. */
  stable: string
  value: string
}

/** Where the surface keeps what it knows between actions (a window global in
 *  the preview page), so a handle still means something on the next call. */
export interface PreviewActHolder {
  /** Target of the action in flight, for the watch overlay to draw onto. */
  aimed?: Element | null
  /** Every handle minted on this page, live or not yet retired. */
  book?: PreviewActBinding[]
  /** Next disambiguating suffix per ref stem, so two "Edit" buttons become
   *  `btn-edit` and `btn-edit-1`. Never rewound: a retired handle's name is
   *  not handed to a different element later in the same page. */
  coined?: Record<string, number>
  /** The on-screen subset, for the overlay to outline. Diverges from `nodes` in
   *  both directions: it drops what is below the fold, and it is not capped at
   *  the inventory's size. */
  field?: Element[]
  nodes?: Element[]
  /** URL the snapshot was taken on; a navigation retires every handle. */
  url?: string
}

/** Run one action against `doc`, resolving refs through `holder`. Self-contained. */
export function actInPage(doc: Document, holder: PreviewActHolder, action: PreviewActAction): PreviewActResult {
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

  /** On screen right now, and worth drawing a box around. The field is strictly
   *  viewport-bound: a mark past the fold is one nobody can see, it spends the
   *  budget that should have gone to what IS on screen, and the ones parked far
   *  off to the side were landing as stray labels in the corner. */
  const onScreen = (el: Element): boolean => {
    if (!win) {
      return false
    }

    const box = el.getBoundingClientRect()

    if (box.right <= 0 || box.bottom <= 0 || box.left >= win.innerWidth || box.top >= win.innerHeight) {
      return false
    }

    // Page-sized wrappers. Outlining one draws a rectangle around the whole
    // screen, which says nothing and frames everything inside it as if it
    // mattered. Full-width banners are fine — it takes both dimensions.
    return !(box.width >= win.innerWidth * 0.95 && box.height >= win.innerHeight * 0.9)
  }

  /** Painted at all, ancestors included. */
  const shown = (el: Element): boolean => {
    // Folds in display/visibility/opacity/content-visibility inherited from an
    // ANCESTOR, which this element's own computed style does not report: inside
    // a parent at opacity 0, every child still reads back opacity 1.
    const seen = (el as Element & { checkVisibility?: (opts: object) => boolean }).checkVisibility

    if (typeof seen === 'function' && !seen.call(el, { checkOpacity: true, checkVisibilityCSS: true })) {
      return false
    }

    const style = win && win.getComputedStyle ? win.getComputedStyle(el) : null

    if (style) {
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
        return false
      }

      // The screen-reader-only recipe, both spellings: the deprecated `clip` and
      // the modern `clip-path: inset(50%)`. Neither exists for any purpose other
      // than hiding something visually while keeping it focusable.
      if ((style.clip && style.clip !== 'auto') || (style.clipPath || '').indexOf('inset(50%') !== -1) {
        return false
      }
    }

    return true
  }

  const visible = (el: Element): boolean => {
    if (!shown(el)) {
      return false
    }

    // Things the page itself declares are not for interacting with, neither of
    // which shows up in a computed style or a bounding box: an aria-hidden
    // subtree is invisible to every other assistive client, and `inert` cannot
    // be clicked at all. Disabled controls are deliberately NOT filtered here —
    // they stay in the inventory so the agent is told a button is disabled
    // rather than hunting for one that appears not to exist.
    if (el.closest('[aria-hidden="true"], [inert]')) {
      return false
    }

    const rect = el.getBoundingClientRect()

    // Below the fold is fine — we scroll to it. Off to the LEFT or ABOVE is the
    // other half of the sr-only trick (`left: -9999px`, `top: -9999px`) and
    // scrolling never brings those back.
    if (rect.right <= 0 || rect.bottom <= 0) {
      return false
    }

    // Bigger than the 1px box sr-only collapses to. Nothing a person can aim at
    // is 2px across, and admitting those is what put the cursor in the corner:
    // they cluster at the document origin, so they sort to the FRONT of the
    // inventory and the agent reaches for one as @e1.
    if (rect.width < 3 || rect.height < 3) {
      return false
    }

    // Occlusion, and the last filter for a reason: everything above is cheap
    // and this one forces layout. If the middle of the element is on screen,
    // whatever the browser reports at that point has to BE this element — its
    // own descendant (an icon inside a link) or the box it paints inside are
    // fine, anything else means it is buried under a sticky header, a cookie
    // wall, or a transparent layer the page put on top. Those are exactly the
    // targets the agent aims at and then misses. Elements below the fold cannot
    // be hit-tested from here, so they are taken on trust and land back in this
    // function after the scroll.
    const midX = rect.left + rect.width / 2
    const midY = rect.top + rect.height / 2

    const under = (doc as Document & { elementFromPoint?: (x: number, y: number) => Element | null })
      .elementFromPoint

    if (
      typeof under === 'function' &&
      win &&
      midX >= 0 &&
      midY >= 0 &&
      midX < win.innerWidth &&
      midY < win.innerHeight
    ) {
      const over = under.call(doc, midX, midY)

      if (!over || !(over === el || el.contains(over) || over.contains(el))) {
        return false
      }
    }

    return true
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
      const role = el.getAttribute('role') || (tag === 'input' ? 'input:' + ((el as HTMLInputElement).type || 'text') : tag)
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

  /** What moved on an element that kept its handle, or nothing if it held
   *  still. Field by field, so a status line ticking over costs the agent one
   *  short line instead of a re-run of everything already known about it. */
  const shifted = (was: PreviewActBinding, entry: PreviewElement): PreviewElementChange | null => {
    const off = !!entry.disabled
    const value = entry.value || ''

    if (was.label === entry.label && was.value === value && was.off === off) {
      return null
    }

    const moved: PreviewElementChange = { ref: was.ref }

    if (was.label !== entry.label) {
      moved.label = entry.label
    }

    if (was.value !== value) {
      moved.value = value
    }

    if (was.off !== off) {
      moved.disabled = off
    }

    return moved
  }

  /** Do two labels share at least half their words? Tolerates the count badge
   *  and the copy edit — "Inbox" against "Inbox (3)". */
  const alike = (a: string, b: string): boolean => {
    if (!a || !b) {
      return false
    }

    const one = a.toLowerCase().split(/\s+/).filter(Boolean)
    const two = b.toLowerCase().split(/\s+/).filter(Boolean)
    const both = one.filter(word => two.indexOf(word) !== -1).length
    const all = one.length + two.filter(word => one.indexOf(word) === -1).length

    return all > 0 && both / all >= 0.5
  }

  /** How strongly a remembered element matches one just observed, 0 to 1.
   *
   *  Ported from anchortree's re-bind ladder (Apache-2.0), minus its geometry
   *  rung: a centroid is only ever worth 0.1 there, it never reaches the 0.6
   *  bar on its own, and carrying coordinates through the book to buy a
   *  tie-break is not worth the measurement. */
  const affinity = (was: PreviewActBinding, now: PreviewActBinding): number => {
    // A button is not a link, however alike the rest of it reads.
    if (was.role !== now.role) {
      return 0
    }

    // Two elements that BOTH carry a stable attribute and disagree are the page
    // telling us outright that they are different things.
    if (was.stable && now.stable) {
      return was.stable === now.stable ? 1 : 0
    }

    let score = 0

    if (was.name && was.name === now.name) {
      score += 0.6
    } else if (alike(was.name, now.name)) {
      score += 0.4
    }

    if (was.path && was.path === now.path) {
      score += 0.3
    }

    return score
  }

  /** Mint a handle. Legible on purpose: the agent reads `btn-sign-in` in a
   *  three-line delta on turn nine and knows what it is, where `@e42` would
   *  send it back to an inventory twenty thousand tokens ago. Suffixes are
   *  never rewound, so a retired handle's name is not later handed to a
   *  different element on the same page. */
  const coin = (role: string, name: string): string => {
    const coined = holder.coined || (holder.coined = {})
    const named = slug(name)
    const stem = stemOf(role) + (named ? '-' + named : '')
    const nth = coined[stem] || 0

    coined[stem] = nth + 1

    return nth ? stem + '-' + nth : stem
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

      now.ref = coin(entry.role, now.name)
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
        return { error: 'The page navigated since the last snapshot, so ' + ref + ' no longer points anywhere. Call elements again.' }
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
