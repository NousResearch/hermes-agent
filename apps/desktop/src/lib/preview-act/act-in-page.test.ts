import { beforeEach, describe, expect, it, vi } from 'vitest'

import { actInPage, type PreviewActHolder } from './act-in-page'

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

beforeEach(() => {
  vi.restoreAllMocks()
  layOutTheDocument()
  Element.prototype.scrollIntoView = vi.fn()
  // jsdom has no hit-testing at all, so the engine skips the occlusion check
  // here by default. Tests that stand one in must not leak it into the next.
  delete (document as Document & { elementFromPoint?: unknown }).elementFromPoint
})

describe('self-containment', () => {
  // The pane injects `actInPage.toString()` into the guest page, where module
  // scope does not exist. A single free identifier (a module-level constant, an
  // imported helper) is a ReferenceError on every call, and Electron reports it
  // only as "Script failed to execute" — so evaluate the source the way the
  // guest does and run a real action through it.
  it('runs after being stringified and eval’d with no module scope', () => {
    const page = document.createElement('div')
    page.innerHTML = '<button id="save">Save</button>'
    document.body.replaceChildren(page)

    const injected = new Function('return (' + actInPage.toString() + ')')() as typeof actInPage
    const holder: PreviewActHolder = {}

    expect(injected(document, holder, { kind: 'elements' }).elements?.[0].label).toBe('Save')

    const clicked = vi.fn()
    document.getElementById('save')!.addEventListener('click', clicked)

    expect(injected(document, holder, { kind: 'click', ref: '@e1' }).success).toBe(true)
    expect(clicked).toHaveBeenCalledOnce()
  })
})

describe('elements', () => {
  it('numbers the interactive nodes with browser_*-style refs', () => {
    const holder = page(`
      <button id="save">Save</button>
      <a href="/help">Help</a>
      <input id="who" placeholder="Your name" />
      <p>Not interactive</p>
    `)

    const result = inventory(holder)

    expect(result.success).toBe(true)
    expect(result.elements?.map(e => [e.ref, e.label])).toEqual([
      ['@e1', 'Save'],
      ['@e2', 'Help'],
      ['@e3', 'Your name']
    ])
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

  it('prefers an identity selector so the agent can re-find the node later', () => {
    const holder = page(`
      <div><button data-testid="submit">Send</button></div>
      <div><button>Plain</button></div>
    `)

    const [byTestId, positional] = inventory(holder).elements!

    expect(byTestId.selector).toBe('[data-testid="submit"]')
    expect(document.querySelector(positional.selector)).toBe(document.querySelectorAll('button')[1])
  })

  it('honours the cap', () => {
    const holder = page(Array.from({ length: 10 }, (_, i) => `<button>B${i}</button>`).join(''))

    expect(inventory(holder).elements).toHaveLength(10)
    expect(actInPage(document, holder, { kind: 'elements', max: 3 }).elements).toHaveLength(3)
  })
})

describe('click', () => {
  it('activates the element a ref points at', () => {
    const holder = page('<button id="save">Save</button>')
    inventory(holder)

    const clicked = vi.fn()
    document.getElementById('save')!.addEventListener('click', clicked)

    const result = actInPage(document, holder, { kind: 'click', ref: '@e1' })

    expect(result.success).toBe(true)
    expect(result.acted).toContain('Save')
    expect(clicked).toHaveBeenCalledOnce()
  })

  it('replays the pointer/mouse sequence frameworks bind to', () => {
    const holder = page('<button id="save">Save</button>')
    inventory(holder)

    const seen: string[] = []

    for (const type of ['pointerdown', 'mousedown', 'mouseup', 'pointerup', 'click']) {
      document.getElementById('save')!.addEventListener(type, () => seen.push(type))
    }

    actInPage(document, holder, { kind: 'click', ref: '@e1' })

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
    inventory(holder)

    const result = actInPage(document, holder, { kind: 'click', ref: '@e1' })

    expect(result.success).toBe(false)
    expect(result.error).toContain('disabled')
  })

  it('reports the live url so a navigation is visible to the agent', () => {
    const holder = page('<button id="save">Save</button>')
    inventory(holder)

    expect(actInPage(document, holder, { kind: 'click', ref: '@e1' }).url).toBe(document.location.href)
  })
})

describe('stale refs', () => {
  it('names an unknown ref rather than clicking whatever sits at that index', () => {
    const holder = page('<button>Only</button>')
    inventory(holder)

    const result = actInPage(document, holder, { kind: 'click', ref: '@e9' })

    expect(result.success).toBe(false)
    expect(result.error).toContain('elements')
  })

  it('catches a node that was removed after the snapshot', () => {
    const holder = page('<button id="save">Save</button>')
    inventory(holder)
    document.getElementById('save')!.remove()

    expect(actInPage(document, holder, { kind: 'click', ref: '@e1' }).error).toContain('removed')
  })

  it('invalidates every ref when the page navigated under them', () => {
    const holder = page('<button>Save</button>')
    inventory(holder)
    holder.url = 'https://elsewhere.example/other'

    expect(actInPage(document, holder, { kind: 'click', ref: '@e1' }).error).toContain('navigated')
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
    inventory(holder)

    const input = document.getElementById('who') as HTMLInputElement
    const events: string[] = []
    input.addEventListener('input', () => events.push('input'))
    input.addEventListener('change', () => events.push('change'))

    const result = actInPage(document, holder, { kind: 'type', ref: '@e1', text: 'Brooklyn' })

    expect(result.success).toBe(true)
    expect(input.value).toBe('Brooklyn')
    expect(events).toEqual(['input', 'change'])
  })

  it('bypasses the own-property shadow React installs on tracked inputs', () => {
    const holder = page('<input id="who" placeholder="Your name" />')
    inventory(holder)

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

    actInPage(document, holder, { kind: 'type', ref: '@e1', text: 'Brooklyn' })

    expect(shadowWrites).toEqual([])
    expect(nativeValue.get!.call(input)).toBe('Brooklyn')
  })

  it('writes into a contenteditable host', () => {
    const holder = page('<div id="editor" contenteditable="true" aria-label="Body"></div>')
    inventory(holder)

    actInPage(document, holder, { kind: 'type', ref: '@e1', text: 'hello' })

    expect(document.getElementById('editor')!.textContent).toBe('hello')
  })

  it('submits the owning form when asked', () => {
    const holder = page('<form id="f"><input id="q" placeholder="Search" /></form>')
    inventory(holder)

    const form = document.getElementById('f') as HTMLFormElement
    form.requestSubmit = vi.fn()

    const result = actInPage(document, holder, { kind: 'type', ref: '@e1', submit: true, text: 'cats' })

    expect(form.requestSubmit).toHaveBeenCalledOnce()
    expect(result.acted).toContain('submitted')
  })

  it('refuses a target that has no text to type into', () => {
    const holder = page('<button id="b">Press</button>')
    inventory(holder)

    expect(actInPage(document, holder, { kind: 'type', ref: '@e1', text: 'x' }).error).toContain('not a text field')
  })
})

describe('press', () => {
  it('sends the key to the target', () => {
    const holder = page('<input id="q" placeholder="Search" />')
    inventory(holder)

    const keys: string[] = []
    document.getElementById('q')!.addEventListener('keydown', e => keys.push((e as KeyboardEvent).key))

    expect(actInPage(document, holder, { key: 'Enter', kind: 'press', ref: '@e1' }).success).toBe(true)
    expect(keys).toEqual(['Enter'])
  })

  it('needs a key', () => {
    const holder = page('<input id="q" placeholder="Search" />')
    inventory(holder)

    expect(actInPage(document, holder, { kind: 'press', ref: '@e1' }).error).toContain('key')
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
    inventory(holder)

    const list = document.getElementById('list') as HTMLElement
    list.scrollBy = vi.fn()
    const pageScroll = vi.spyOn(window, 'scrollBy').mockImplementation(() => {})

    const result = actInPage(document, holder, { amount: 200, kind: 'scroll', ref: '@e1' })

    expect(list.scrollBy).toHaveBeenCalledWith({ behavior: 'smooth', top: 200 })
    expect(pageScroll).not.toHaveBeenCalled()
    expect(result.acted).toContain('Results')
  })
})
