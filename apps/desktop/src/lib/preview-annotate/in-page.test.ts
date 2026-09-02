import { afterEach, describe, expect, it } from 'vitest'

import { annotateInPage } from './in-page'

afterEach(() => {
  document.body.replaceChildren()
  document.querySelectorAll('hermes-annotate').forEach(node => node.remove())
})

describe('annotateInPage overlay', () => {
  it('installs a host and tears it down completely', () => {
    const api = annotateInPage(document)
    api.install()

    expect(api.isInstalled()).toBe(true)
    expect(document.querySelector('hermes-annotate')).toBeTruthy()

    api.teardown()

    expect(api.isInstalled()).toBe(false)
    expect(document.querySelector('hermes-annotate')).toBeNull()
  })

  it('paints a blue outline and numbered markers for stacked pins', () => {
    const api = annotateInPage(document)
    api.install()
    api.showPins([
      { kind: 'element', number: 1, rect: { height: 24, width: 80, x: 10, y: 10 } },
      { kind: 'area', number: 2, rect: { height: 40, width: 120, x: 20, y: 80 } }
    ])

    expect(api.getMarkerNumbers()).toEqual([1, 2])
    expect(api.getOutlineColor().replace(/\s/g, '').toLowerCase()).toMatch(/#2f80ed|rgb\(47,128,237\)/)
  })

  it('keeps a pin glued to its element after the element moves', () => {
    const word = document.createElement('span')
    word.id = 'target-word'
    word.textContent = 'hello'
    document.body.appendChild(word)
    let top = 40
    word.getBoundingClientRect = () => ({
      bottom: top + 20,
      height: 20,
      left: 10,
      right: 90,
      toJSON: () => ({}),
      top,
      width: 80,
      x: 10,
      y: top
    })

    const api = annotateInPage(document)
    api.install()
    api.showPins([
      { kind: 'element', number: 1, rect: { height: 20, width: 80, x: 10, y: 40 }, selector: '#target-word' }
    ])

    const pin = () =>
      document.querySelector('hermes-annotate')!.shadowRoot!.querySelector('[data-annotate-pin="1"]') as HTMLElement

    expect(pin().style.top).toBe('40px')
    top = 200
    api.relocate()
    expect(pin().style.top).toBe('200px')
    api.teardown()
  })

  it('keeps the picked element when the draft becomes a pin, even if the selector is wrong', () => {
    const word = document.createElement('span')
    word.id = 'target-word'
    word.textContent = 'hello'
    document.body.appendChild(word)
    let top = 40
    word.getBoundingClientRect = () => ({
      bottom: top + 20,
      height: 20,
      left: 10,
      right: 90,
      toJSON: () => ({}),
      top,
      width: 80,
      x: 10,
      y: top
    })
    document.elementFromPoint = () => word

    const api = annotateInPage(document)
    api.install()
    const host = document.querySelector('hermes-annotate') as HTMLElement
    host.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 20, clientY: 48 }))
    host.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, button: 0, clientX: 21, clientY: 49 }))

    api.showPins([
      { kind: 'element', number: 1, rect: { height: 20, width: 80, x: 10, y: 40 }, selector: '#does-not-exist' }
    ])

    const pin = () =>
      document.querySelector('hermes-annotate')!.shadowRoot!.querySelector('[data-annotate-pin="1"]') as HTMLElement

    expect(pin().style.top).toBe('40px')
    top = 180
    api.relocate()
    expect(pin().style.top).toBe('180px')
    api.teardown()
  })

  it('emits pick-element with compact identity on click', async () => {
    const button = document.createElement('button')
    button.id = 'go'
    button.textContent = 'Go'
    document.body.appendChild(button)
    button.getBoundingClientRect = () => ({
      bottom: 34,
      height: 24,
      left: 10,
      right: 90,
      toJSON: () => ({}),
      top: 10,
      width: 80,
      x: 10,
      y: 10
    })
    document.elementFromPoint = () => button

    const api = annotateInPage(document)
    api.install()
    const pending = api.wait()
    const host = document.querySelector('hermes-annotate') as HTMLElement

    host.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 20, clientY: 20 }))
    host.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, button: 0, clientX: 21, clientY: 21 }))

    const event = await pending

    expect(event.type).toBe('pick-element')

    if (event.type === 'pick-element') {
      expect(event.identity.tag).toBe('button')
      expect(event.identity.selector).toContain('go')
      expect(event.identity.text).toBe('Go')
      expect(event.identity.html).toContain('<button')
      expect(event.identity.html).toContain('Go')
    }
  })

  it('redacts secret-shaped values before the markup leaves the page', async () => {
    const form = document.createElement('form')
    form.setAttribute('data-api-key', 'sk-live-1234567890')
    form.innerHTML =
      '<input type="password" value="hunter2"><input type="hidden" name="csrf" value="tok_abc">' +
      '<input type="email" value="me@example.com">'
    document.body.appendChild(form)
    form.getBoundingClientRect = () => ({
      bottom: 60,
      height: 50,
      left: 0,
      right: 200,
      toJSON: () => ({}),
      top: 10,
      width: 200,
      x: 0,
      y: 10
    })
    document.elementFromPoint = () => form

    const api = annotateInPage(document)
    api.install()
    const pending = api.wait()
    const host = document.querySelector('hermes-annotate') as HTMLElement

    host.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 20, clientY: 20 }))
    host.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, button: 0, clientX: 21, clientY: 21 }))

    const event = await pending

    expect(event.type).toBe('pick-element')

    if (event.type === 'pick-element') {
      expect(event.identity.html).not.toContain('hunter2')
      expect(event.identity.html).not.toContain('tok_abc')
      expect(event.identity.html).not.toContain('sk-live-1234567890')
      expect(event.identity.html).toContain('[redacted]')
      // A non-secret field keeps its value — redaction is targeted, not a blanket wipe.
      expect(event.identity.html).toContain('me@example.com')
    }

    api.teardown()
  })

  it('budgets the markup so one comment cannot paste a whole section', async () => {
    const section = document.createElement('section')
    section.innerHTML = '<p>filler filler filler</p>'.repeat(200)
    document.body.appendChild(section)
    section.getBoundingClientRect = () => ({
      bottom: 400,
      height: 400,
      left: 0,
      right: 300,
      toJSON: () => ({}),
      top: 0,
      width: 300,
      x: 0,
      y: 0
    })
    document.elementFromPoint = () => section

    const api = annotateInPage(document)
    api.install()
    const pending = api.wait()
    const host = document.querySelector('hermes-annotate') as HTMLElement

    host.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 20, clientY: 20 }))
    host.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, button: 0, clientX: 21, clientY: 21 }))

    const event = await pending

    if (event.type === 'pick-element') {
      expect(event.identity.html.length).toBeLessThanOrEqual(600)
      expect(event.identity.html.startsWith('<section>')).toBe(true)
    }

    api.teardown()
  })

  it('shows the draft marker before a capture can photograph it', async () => {
    const api = annotateInPage(document)
    api.install()
    api.showDraft({ height: 24, width: 80, x: 10, y: 10 }, 2)

    const ready = await api.beginCapture()
    const shadow = document.querySelector('hermes-annotate')!.shadowRoot!
    const draft = shadow.querySelector('[data-annotate-outline="draft"]') as HTMLElement
    const marker = draft.querySelector('[data-annotate-marker]')

    expect(ready).toBe(true)
    expect(draft.style.display).toBe('block')
    expect(marker?.getAttribute('data-annotate-marker')).toBe('2')
    api.teardown()
  })

  it('hides saved pins during a capture so a neighbour marker cannot land in the crop', async () => {
    const api = annotateInPage(document)
    api.install()
    api.showPins([
      { kind: 'element', number: 1, rect: { height: 24, width: 80, x: 10, y: 10 } },
      { kind: 'element', number: 2, rect: { height: 24, width: 80, x: 10, y: 40 } }
    ])
    api.showDraft({ height: 24, width: 80, x: 10, y: 70 }, 3)

    const shadow = document.querySelector('hermes-annotate')!.shadowRoot!
    const pins = () => shadow.querySelector('[data-annotate-pin="1"]')!.parentElement as HTMLElement

    expect(pins().style.display).not.toBe('none')

    await api.beginCapture()

    expect(pins().style.display).toBe('none')
    // The draft's own marker must survive — it is the point of the crop.
    expect(
      (shadow.querySelector('[data-annotate-outline="draft"]') as HTMLElement).style.display
    ).toBe('block')

    api.endCapture()

    expect(pins().style.display).toBe('block')
    api.teardown()
  })

  it('refuses to dress a capture when the overlay is gone', async () => {
    const api = annotateInPage(document)

    expect(await api.beginCapture()).toBe(false)
  })

  it('owns wheel scrolling instead of also allowing the native wheel action', () => {
    const scroller = document.createElement('div')
    scroller.style.overflowY = 'auto'
    Object.defineProperties(scroller, {
      clientHeight: { configurable: true, value: 100 },
      scrollHeight: { configurable: true, value: 500 }
    })
    document.body.appendChild(scroller)
    document.elementFromPoint = () => scroller

    const api = annotateInPage(document)
    api.install()
    const host = document.querySelector('hermes-annotate') as HTMLElement
    const wheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 30 })

    host.dispatchEvent(wheel)

    expect(wheel.defaultPrevented).toBe(true)
    expect(scroller.scrollTop).toBe(30)
    api.teardown()
  })
})
