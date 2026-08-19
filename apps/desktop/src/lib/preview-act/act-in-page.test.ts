import { beforeEach, describe, expect, it, vi } from 'vitest'

import { actInPage, type PreviewActHolder } from './act-in-page'

/** jsdom lays nothing out, so every rect is 0×0 and the engine's visibility
 *  check would reject the whole page. Give elements a plausible box and let
 *  `display: none` (which jsdom DOES compute) carry the hiding. */
function layOutTheDocument() {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const hidden = getComputedStyle(this).display === 'none'
    const size = hidden ? 0 : 40

    return { bottom: size, height: size, left: 0, right: size, top: 0, width: size, x: 0, y: 0 } as DOMRect
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
    expect(scrollBy).toHaveBeenCalledWith({ behavior: 'auto', top: Math.round(window.innerHeight * 0.9) })
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

    expect(list.scrollBy).toHaveBeenCalledWith({ behavior: 'auto', top: 200 })
    expect(pageScroll).not.toHaveBeenCalled()
    expect(result.acted).toContain('Results')
  })
})
