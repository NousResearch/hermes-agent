import { beforeEach, describe, expect, it, vi } from 'vitest'

import { actEngineSource, actInPage, type PreviewActHolder } from './act-in-page'

/** jsdom lays nothing out, so every rect is 0×0 and the engine's visibility
 *  check would reject the whole page. Give elements a plausible box, honouring
 *  an explicit width/height so a test can lay out a 1px one, and let
 *  `display: none` (which jsdom DOES compute) carry the hiding. */
function layOutTheDocument() {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const style = getComputedStyle(this)
    const width = style.display === 'none' ? 0 : parseFloat(style.width) || 40
    const height = style.display === 'none' ? 0 : parseFloat(style.height) || 40
    const left = parseFloat(style.left) || 0

    return { bottom: height, height, left, right: left + width, top: 0, width, x: left, y: 0 } as DOMRect
  })
}

function page(html: string): PreviewActHolder {
  document.body.innerHTML = html

  return {}
}

/** Take the inventory the agent would take before acting. */
function inventory(holder: PreviewActHolder) {
  return actInPage(document, holder, { kind: 'elements' })
}

/** Take the inventory and hand back the handle for the first thing on the page.
 *  Tests address elements the way the agent does — by asking what they are
 *  called — rather than predicting what the engine will name them. */
function firstRef(holder: PreviewActHolder) {
  return inventory(holder).elements![0].ref
}

beforeEach(() => {
  vi.restoreAllMocks()
  layOutTheDocument()
  Element.prototype.scrollIntoView = vi.fn()
  // jsdom has no hit-testing at all, so the engine skips the occlusion check
  // here by default. Tests that stand one in must not leak it into the next.
  delete (document as unknown as { elementFromPoint?: unknown }).elementFromPoint
})

describe('self-containment', () => {
  // The pane injects `actEngineSource()` into the guest page, where module
  // scope does not exist. A single free identifier — a module-level constant,
  // an imported helper, or a kit factory the assembler forgot to ship — is a
  // ReferenceError on every call, and Electron reports it only as "Script
  // failed to execute". Evaluating the bundle the way the guest does is the
  // only way to catch that here: a plain import of these functions resolves
  // the very names the guest would be missing.
  //
  // It is also the guard on the build. The renderer minifies, so an imported
  // binding named inside the stringified core would be renamed to something
  // the bundle never declares — green in dev and in this suite if the source
  // were assembled any other way, broken only once packaged.
  it('runs after being stringified and eval’d with no module scope', () => {
    const page = document.createElement('div')
    page.innerHTML = '<button id="save">Save</button>'
    document.body.replaceChildren(page)

    const injected = new Function('return ' + actEngineSource())() as typeof actInPage
    const holder: PreviewActHolder = {}

    const [save] = injected(document, holder, { kind: 'elements' }).elements!

    expect(save.label).toBe('Save')

    const clicked = vi.fn()
    document.getElementById('save')!.addEventListener('click', clicked)

    expect(injected(document, holder, { kind: 'click', ref: save.ref }).success).toBe(true)
    expect(clicked).toHaveBeenCalledOnce()
  })
})

describe('elements', () => {
  // The handle says what the thing is and which one it is, so a delta line the
  // agent reads ten turns later needs no lookup to make sense of.
  it('names the interactive nodes after their role and label', () => {
    const holder = page(`
      <button id="save">Save</button>
      <a href="/help">Help</a>
      <input id="who" placeholder="Your name" />
      <p>Not interactive</p>
    `)

    const result = inventory(holder)

    expect(result.success).toBe(true)
    expect(result.elements?.map(e => [e.ref, e.label])).toEqual([
      ['btn-save', 'Save'],
      ['lnk-help', 'Help'],
      ['inp-your-name', 'Your name']
    ])
  })

  it('tells two elements with the same name apart', () => {
    const holder = page(`
      <button>Edit</button>
      <button>Edit</button>
      <button>Edit</button>
    `)

    expect(inventory(holder).elements?.map(e => e.ref)).toEqual(['btn-edit', 'btn-edit-1', 'btn-edit-2'])
  })

  // Handing a retired name to a different element would silently redirect a
  // handle the agent is still holding. The two ids here also disagree, which is
  // the page saying outright that these are different buttons — so this must
  // not re-bind despite the identical label.
  it('never reissues the name of an element that went away', () => {
    const holder = page(`
      <div id="host"><button id="a">Edit</button></div>
      <a href="/help">Help</a>
      <a href="/terms">Terms</a>
    `)

    expect(firstRef(holder)).toBe('btn-edit')

    document.getElementById('host')!.innerHTML = '<button id="b">Edit</button>'

    const again = inventory(holder)

    expect(again.delta?.removed).toEqual(['btn-edit'])
    expect(again.delta?.added?.map(e => e.ref)).toEqual(['btn-edit-1'])
  })

  it('reports role, current value, and disabled state', () => {
    const holder = page(`
      <input id="email" aria-label="Email" type="email" value="a@b.co" />
      <button id="go" disabled>Go</button>
    `)

    const [email, go] = inventory(holder).elements!

    expect(email).toMatchObject({ label: 'Email', role: 'input:email', value: 'a@b.co' })
    expect(go).toMatchObject({ disabled: true, label: 'Go' })
    expect(email.disabled).toBeUndefined()
  })

  it('skips hidden controls and unlabelled ones', () => {
    const holder = page(`
      <button id="real">Real</button>
      <button id="gone" style="display: none">Gone</button>
      <button id="mystery"></button>
    `)

    expect(inventory(holder).elements?.map(e => e.label)).toEqual(['Real'])
  })

  // The screen-reader-only recipe is a 1px box parked at the document origin,
  // and it passed a `width >= 1` check. Skip links and CSS-only menu toggles are
  // built this way and come FIRST in the document, so they landed on @e1 — the
  // agent would aim at one and the pointer would fly to the top-left corner and
  // click nothing.
  it('skips the visually-hidden controls that sit at the document origin', () => {
    const holder = page(`
      <a href="#content" style="width: 1px; height: 1px; clip: rect(0, 0, 0, 0)">Jump to content</a>
      <input aria-label="Toggle sidebar" style="width: 1px; height: 1px" type="checkbox" />
      <a href="#main" style="clip-path: inset(50%)">Skip navigation</a>
      <button id="real">Search</button>
    `)

    expect(inventory(holder).elements?.map(e => e.label)).toEqual(['Search'])
  })

  it('skips a control parked off the left edge, which no scroll brings back', () => {
    const holder = page(`
      <button style="position: absolute; left: -9999px">Hidden</button>
      <button id="real">Search</button>
    `)

    expect(inventory(holder).elements?.map(e => e.label)).toEqual(['Search'])
  })

  // Wikipedia and friends are full of these: decorative chrome and collapsed
  // menus that are perfectly solid boxes as far as layout is concerned, but
  // that the page has already declared are not for anyone to interact with.
  it('skips controls the page marked aria-hidden or inert', () => {
    const holder = page(`
      <div aria-hidden="true"><button>Decoration</button></div>
      <div inert><button>Collapsed menu</button></div>
      <button id="real">Search</button>
    `)

    expect(inventory(holder).elements?.map(e => e.label)).toEqual(['Search'])
  })

  it('skips a control buried under another layer, which would take the click instead', () => {
    const holder = page(`
      <button style="left: 100px">Accept cookies</button>
      <button id="real" style="left: 300px">Search</button>
    `)

    const wall = document.createElement('div')
    document.body.append(wall)

    // Stand in for the hit-testing jsdom does not do: every element reports
    // itself except the buried one, which reports the sheet lying over it.
    document.elementFromPoint = (x: number) => (x === 120 ? wall : document.getElementById('real'))

    expect(inventory(holder).elements?.map(e => e.label)).toEqual(['Search'])
  })

  it('keeps a control whose own child is what the hit test lands on', () => {
    const holder = page('<a href="/home" id="real"><svg id="icon"></svg> Home</a>')

    document.elementFromPoint = () => document.getElementById('icon')

    expect(inventory(holder).elements?.map(e => e.label)).toEqual(['Home'])
  })

  // The overlay draws far more than the agent is told about, so the two lists
  // are collected separately: the field is every visible control, the inventory
  // is the describable subset that refs point into.
  it('parks the whole interactive field for the overlay, not just the inventory', () => {
    const holder = page(`
      <button id="named">Search</button>
      <button id="mystery"></button>
      <button id="gone" style="display: none">Gone</button>
    `)

    expect(inventory(holder).elements?.map(e => e.label)).toEqual(['Search'])
    expect(holder.nodes).toEqual([document.getElementById('named')])
    expect(holder.field).toEqual([document.getElementById('named'), document.getElementById('mystery')])
  })

  // The positional `:nth-child` chain this used to fall back to was 74% of a
  // real page's inventory and nothing read it. An identity selector is short
  // and stable, so it stays; everything else addresses by ref.
  it('carries an identity selector only, and omits it when there is none', () => {
    const holder = page(`
      <div><button data-testid="submit">Send</button></div>
      <div><button id="cancel">Cancel</button></div>
      <div><button>Plain</button></div>
    `)

    const [byTestId, byId, plain] = inventory(holder).elements!

    expect(byTestId.selector).toBe('[data-testid="submit"]')
    expect(byId.selector).toBe('#cancel')
    expect(plain.selector).toBeUndefined()
    expect(plain.ref).toBe('btn-plain')
  })

  it('honours the cap', () => {
    const holder = page(Array.from({ length: 10 }, (_, i) => `<button>B${i}</button>`).join(''))

    expect(inventory(holder).elements).toHaveLength(10)
    expect(actInPage(document, holder, { full: true, kind: 'elements', max: 3 }).elements).toHaveLength(3)
  })
})

// Re-sending the whole inventory after every click is what made a ten-step
// session cost several times what it needed to: the page barely moves between
// steps and the agent was charged for a fresh copy of it each time.
describe('delta', () => {
  it('gives the full inventory the first time it looks at a page', () => {
    const holder = page('<button>Save</button>')
    const first = inventory(holder)

    expect(first.elements).toHaveLength(1)
    expect(first.delta).toBeUndefined()
  })

  it('reports only what moved on every look after that', () => {
    const holder = page(`
      <div id="host">
        <button id="save">Save</button>
        <button id="undo">Undo</button>
        <button id="redo">Redo</button>
      </div>
    `)

    inventory(holder)

    document.getElementById('host')!.insertAdjacentHTML('beforeend', '<button id="quit">Quit</button>')

    const next = inventory(holder)

    expect(next.elements).toBeUndefined()
    expect(next.delta?.added?.map(e => e.ref)).toEqual(['btn-quit'])
    expect(next.delta?.same).toBe(3)
  })

  it('says nothing about a page that did not move', () => {
    const holder = page('<button id="save">Save</button>')
    inventory(holder)

    const still = inventory(holder)

    expect(still.delta).toEqual({ same: 1 })
  })

  it('reports a relabelled control as changed, keeping its handle', () => {
    const holder = page(`
      <button id="cart">Add to cart</button>
      <a href="/help">Help</a>
      <a href="/terms">Terms</a>
    `)

    const ref = firstRef(holder)

    document.getElementById('cart')!.textContent = 'Added'

    const next = inventory(holder)

    expect(next.delta?.changed?.map(e => [e.ref, e.label])).toEqual([[ref, 'Added']])
    expect(actInPage(document, holder, { kind: 'click', ref }).success).toBe(true)
  })

  // `changed` fires on nearly every step of a long task, so it is the one part
  // of the payload whose cost compounds. It carries the moved field and the
  // handle, and nothing the agent already knows.
  it('reports only the field that moved, not the whole element', () => {
    const holder = page(`
      <input id="q" aria-label="Search" />
      <a href="/help">Help</a>
      <a href="/terms">Terms</a>
    `)

    inventory(holder)
    ;(document.getElementById('q') as HTMLInputElement).value = 'shoes'

    const [moved] = inventory(holder).delta!.changed!

    expect(moved).toEqual({ ref: 'inp-search', value: 'shoes' })
  })

  it('reports a control becoming available on its own', () => {
    const holder = page(`
      <button id="go" disabled>Continue</button>
      <a href="/help">Help</a>
      <a href="/terms">Terms</a>
    `)

    inventory(holder)
    ;(document.getElementById('go') as HTMLButtonElement).disabled = false

    expect(inventory(holder).delta?.changed).toEqual([{ disabled: false, ref: 'btn-continue' }])
  })

  it('falls back to the whole inventory when most of the page is new', () => {
    const holder = page('<div id="host"><button id="save">Save</button></div>')
    inventory(holder)

    document.getElementById('host')!.innerHTML = '<a href="/a">A</a><a href="/b">B</a><a href="/c">C</a>'

    const next = inventory(holder)

    expect(next.delta).toBeUndefined()
    expect(next.elements?.map(e => e.ref)).toEqual(['lnk-a', 'lnk-b', 'lnk-c'])
  })

  // An element that slid past the cap is still clickable, so calling it removed
  // would be a lie the agent acts on.
  it('does not report an element as removed just because it fell past the cap', () => {
    const holder = page(Array.from({ length: 4 }, (_, i) => `<button>B${i}</button>`).join(''))
    inventory(holder)

    const capped = actInPage(document, holder, { kind: 'elements', max: 2 })

    expect(capped.delta?.removed).toBeUndefined()
    expect(actInPage(document, holder, { kind: 'click', ref: 'btn-b3' }).success).toBe(true)
  })

  // The contract the whole delta exists for. Not a fixed byte count — that
  // would break on any wording change — but the relationship between the two
  // payloads, which is what has to hold.
  it('costs a fraction of the inventory on a page that mostly held still', () => {
    const holder = page(`
      <nav>${Array.from({ length: 8 }, (_, i) => `<a href="/n${i}">Section ${i}</a>`).join('')}</nav>
      <main>
        ${Array.from({ length: 24 }, (_, i) => `<button id="row-${i}">Row action ${i}</button>`).join('')}
        <input id="q" placeholder="Search everything" />
        <div id="host"></div>
      </main>
    `)

    const baseline = JSON.stringify(inventory(holder).elements)

    document.getElementById('host')!.innerHTML = '<button id="toast">Dismiss</button>'
    document.getElementById('row-3')!.textContent = 'Row action 3 (done)'

    const next = inventory(holder)

    expect(next.delta?.same).toBe(32)
    // Measured at roughly 15x on this fixture; the bar is set well below that
    // so a wording change does not fail the build, but a regression to
    // re-sending the page would.
    expect(JSON.stringify(next.delta).length * 5).toBeLessThan(baseline.length)
  })

  it('retires every handle when the page navigates', () => {
    const holder = page('<button id="save">Save</button>')
    inventory(holder)
    holder.url = 'https://elsewhere.example/other'

    const landed = inventory(holder)

    expect(landed.delta).toBeUndefined()
    expect(landed.elements?.map(e => e.ref)).toEqual(['btn-save'])
  })
})

// The headline case. A framework re-render destroys the node and builds a new
// one; the agent's handle has to survive that, and it has to hear about it in
// one word rather than as a removal it must react to plus an addition it must
// re-read.
describe('rebind', () => {
  /** A page with a stable nav around the part that re-renders, so the re-bind
   *  has to pick its candidate rather than being handed the only one going. */
  function app(inner: string) {
    return page(`
      <nav><a href="/">Home</a><a href="/docs">Docs</a><a href="/pricing">Pricing</a></nav>
      <main id="host">${inner}</main>
    `)
  }

  function rerender(inner: string) {
    document.getElementById('host')!.innerHTML = inner
  }

  it('keeps the handle when a re-render replaces the node', () => {
    const holder = app('<button>Sign in</button>')
    const ref = inventory(holder).elements!.find(e => e.label === 'Sign in')!.ref
    const before = document.querySelector('button')

    rerender('<span><button>Sign in</button></span>')

    const next = inventory(holder)

    expect(document.querySelector('button')).not.toBe(before)
    expect(next.delta?.rebound).toEqual([ref])
    expect(next.delta?.added).toBeUndefined()
    expect(next.delta?.removed).toBeUndefined()
    expect(actInPage(document, holder, { kind: 'click', ref }).success).toBe(true)
  })

  it('follows a label through a count badge appearing on it', () => {
    const holder = app('<a href="/in">Inbox</a>')
    const ref = inventory(holder).elements!.find(e => e.label === 'Inbox')!.ref

    rerender('<div><a href="/in">Inbox (3)</a></div>')

    expect(inventory(holder).delta?.rebound).toEqual([ref])
  })

  it('will not move a handle across roles', () => {
    const holder = app('<button>Continue</button>')
    inventory(holder)

    rerender('<a href="/next">Continue</a>')

    const next = inventory(holder)

    expect(next.delta?.rebound).toBeUndefined()
    expect(next.delta?.removed).toEqual(['btn-continue'])
    expect(next.delta?.added?.map(e => e.ref)).toEqual(['lnk-continue'])
  })

  // Two unrelated buttons trading places must not trade handles with them.
  it('mints a new handle rather than guess between unrelated candidates', () => {
    const holder = app('<button>Delete account</button>')
    inventory(holder)

    rerender('<button>Upload photo</button>')

    const next = inventory(holder)

    expect(next.delta?.rebound).toBeUndefined()
    expect(next.delta?.removed).toEqual(['btn-delete-account'])
    expect(next.delta?.added?.map(e => e.ref)).toEqual(['btn-upload-photo'])
  })

  // Both carry an id and the ids disagree, which is the page saying outright
  // that these are two different controls however alike they read.
  it('will not move a handle between elements the page marks as distinct', () => {
    const holder = app('<button id="save-draft">Save</button>')
    inventory(holder)

    rerender('<button id="save-final">Save</button>')

    const next = inventory(holder)

    expect(next.delta?.rebound).toBeUndefined()
    expect(next.delta?.removed).toEqual(['btn-save'])
  })
})

describe('click', () => {
  it('activates the element a ref points at', () => {
    const holder = page('<button id="save">Save</button>')
    const ref = firstRef(holder)

    const clicked = vi.fn()
    document.getElementById('save')!.addEventListener('click', clicked)

    const result = actInPage(document, holder, { kind: 'click', ref })

    expect(result.success).toBe(true)
    expect(result.acted).toContain('Save')
    expect(clicked).toHaveBeenCalledOnce()
  })

  it('replays the pointer/mouse sequence frameworks bind to', () => {
    const holder = page('<button id="save">Save</button>')
    const ref = firstRef(holder)

    const seen: string[] = []

    for (const type of ['pointerdown', 'mousedown', 'mouseup', 'pointerup', 'click']) {
      document.getElementById('save')!.addEventListener(type, () => seen.push(type))
    }

    actInPage(document, holder, { kind: 'click', ref })

    expect(seen).toContain('mousedown')
    expect(seen).toContain('mouseup')
    expect(seen.at(-1)).toBe('click')
  })

  it('takes a raw CSS selector when no ref fits', () => {
    const holder = page('<button id="save">Save</button>')
    const clicked = vi.fn()
    document.getElementById('save')!.addEventListener('click', clicked)

    expect(actInPage(document, holder, { kind: 'click', selector: '#save' }).success).toBe(true)
    expect(clicked).toHaveBeenCalledOnce()
  })

  it('refuses a disabled control instead of silently doing nothing', () => {
    const holder = page('<button id="save" disabled>Save</button>')
    const result = actInPage(document, holder, { kind: 'click', ref: firstRef(holder) })

    expect(result.success).toBe(false)
    expect(result.error).toContain('disabled')
  })

  it('reports the live url so a navigation is visible to the agent', () => {
    const holder = page('<button id="save">Save</button>')

    expect(actInPage(document, holder, { kind: 'click', ref: firstRef(holder) }).url).toBe(document.location.href)
  })
})

describe('stale refs', () => {
  it('names an unknown ref rather than acting on whatever is nearby', () => {
    const holder = page('<button>Only</button>')
    inventory(holder)

    const result = actInPage(document, holder, { kind: 'click', ref: 'btn-imaginary' })

    expect(result.success).toBe(false)
    expect(result.error).toContain('elements')
  })

  it('catches a node that was removed after the snapshot', () => {
    const holder = page('<button id="save">Save</button>')
    const ref = firstRef(holder)
    document.getElementById('save')!.remove()

    expect(actInPage(document, holder, { kind: 'click', ref }).error).toContain('removed')
  })

  it('invalidates every ref when the page navigated under them', () => {
    const holder = page('<button>Save</button>')
    const ref = firstRef(holder)
    holder.url = 'https://elsewhere.example/other'

    expect(actInPage(document, holder, { kind: 'click', ref }).error).toContain('navigated')
  })

  it('asks for a target when given neither', () => {
    expect(actInPage(document, page('<button>Save</button>'), { kind: 'click' }).error).toContain('selector')
  })

  it('reports a selector that matches nothing', () => {
    expect(actInPage(document, page(''), { kind: 'click', selector: '#nope' }).error).toContain('No element')
  })
})

describe('type', () => {
  it('enters text and fires the events a controlled input listens for', () => {
    const holder = page('<input id="who" placeholder="Your name" />')
    const ref = firstRef(holder)

    const input = document.getElementById('who') as HTMLInputElement
    const events: string[] = []
    input.addEventListener('input', () => events.push('input'))
    input.addEventListener('change', () => events.push('change'))

    const result = actInPage(document, holder, { kind: 'type', ref, text: 'Brooklyn' })

    expect(result.success).toBe(true)
    expect(input.value).toBe('Brooklyn')
    expect(events).toEqual(['input', 'change'])
  })

  it('bypasses the own-property shadow React installs on tracked inputs', () => {
    const holder = page('<input id="who" placeholder="Your name" />')
    const ref = firstRef(holder)

    // React defines its own `value` accessor on the node to track what it last
    // wrote, and ignores an input event that agrees with it. Writing through
    // that shadow is exactly how typed text snaps back on the next render.
    const input = document.getElementById('who') as HTMLInputElement
    const nativeValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!
    const shadowWrites: string[] = []

    Object.defineProperty(input, 'value', {
      configurable: true,
      get: () => nativeValue.get!.call(input),
      set: (next: string) => {
        shadowWrites.push(next)
      }
    })

    actInPage(document, holder, { kind: 'type', ref, text: 'Brooklyn' })

    expect(shadowWrites).toEqual([])
    expect(nativeValue.get!.call(input)).toBe('Brooklyn')
  })

  it('writes into a contenteditable host', () => {
    const holder = page('<div id="editor" contenteditable="true" aria-label="Body"></div>')

    actInPage(document, holder, { kind: 'type', ref: firstRef(holder), text: 'hello' })

    expect(document.getElementById('editor')!.textContent).toBe('hello')
  })

  it('submits the owning form when asked', () => {
    const holder = page('<form id="f"><input id="q" placeholder="Search" /></form>')
    const ref = firstRef(holder)

    const form = document.getElementById('f') as HTMLFormElement
    form.requestSubmit = vi.fn()

    const result = actInPage(document, holder, { kind: 'type', ref, submit: true, text: 'cats' })

    expect(form.requestSubmit).toHaveBeenCalledOnce()
    expect(result.acted).toContain('submitted')
  })

  it('refuses a target that has no text to type into', () => {
    const holder = page('<button id="b">Press</button>')

    expect(actInPage(document, holder, { kind: 'type', ref: firstRef(holder), text: 'x' }).error).toContain(
      'not a text field'
    )
  })
})

describe('press', () => {
  it('sends the key to the target', () => {
    const holder = page('<input id="q" placeholder="Search" />')
    const ref = firstRef(holder)

    const keys: string[] = []
    document.getElementById('q')!.addEventListener('keydown', e => keys.push((e as KeyboardEvent).key))

    expect(actInPage(document, holder, { key: 'Enter', kind: 'press', ref }).success).toBe(true)
    expect(keys).toEqual(['Enter'])
  })

  it('needs a key', () => {
    const holder = page('<input id="q" placeholder="Search" />')

    expect(actInPage(document, holder, { kind: 'press', ref: firstRef(holder) }).error).toContain('key')
  })
})

describe('scroll', () => {
  it('scrolls the page by about a screen when given no distance', () => {
    const holder = page('<p>long page</p>')
    const scrollBy = vi.spyOn(window, 'scrollBy').mockImplementation(() => {})

    const result = actInPage(document, holder, { kind: 'scroll' })

    expect(result.success).toBe(true)
    expect(scrollBy).toHaveBeenCalledWith({ behavior: 'smooth', top: Math.round(window.innerHeight * 0.9) })
  })

  it('drops the animation for a reader who asked for reduced motion', () => {
    const holder = page('<p>long page</p>')
    const scrollBy = vi.spyOn(window, 'scrollBy').mockImplementation(() => {})

    // Passing 'smooth' explicitly overrides the OS preference, so the engine has
    // to opt back out itself rather than leaving it to the browser.
    vi.stubGlobal('matchMedia', () => ({ matches: true }))

    actInPage(document, holder, { kind: 'scroll' })
    vi.unstubAllGlobals()

    const [options] = scrollBy.mock.calls[0] as unknown as [ScrollToOptions]

    expect(options.behavior).toBe('auto')
  })

  it('jumps to the bottom', () => {
    const holder = page('<p>long page</p>')
    const scrollBy = vi.spyOn(window, 'scrollBy').mockImplementation(() => {})

    actInPage(document, holder, { kind: 'scroll', to: 'bottom' })

    const [options] = scrollBy.mock.calls[0] as unknown as [ScrollToOptions]

    expect(options.top).toBeGreaterThan(window.innerHeight)
  })

  it('scrolls a ref’d container instead of the page', () => {
    const holder = page('<div id="list" aria-label="Results" tabindex="0" style="overflow: auto"></div>')
    const ref = firstRef(holder)

    const list = document.getElementById('list') as HTMLElement
    list.scrollBy = vi.fn()
    const pageScroll = vi.spyOn(window, 'scrollBy').mockImplementation(() => {})

    const result = actInPage(document, holder, { amount: 200, kind: 'scroll', ref })

    expect(list.scrollBy).toHaveBeenCalledWith({ behavior: 'smooth', top: 200 })
    expect(pageScroll).not.toHaveBeenCalled()
    expect(result.acted).toContain('Results')
  })
})
