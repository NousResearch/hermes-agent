import { parsePreviewDesignBridgeEvent, parsePreviewDesignCommand } from '@hermes/shared/preview-design'
import { describe, expect, it } from 'vitest'

import { designInPage } from './preview-design-in-page'

describe('Design Mode bridge protocol', () => {
  it('accepts exact bounded messages and rejects extra or stale fields', () => {
    expect(parsePreviewDesignCommand({ session: 1, type: 'start' })).toEqual({ session: 1, type: 'start' })
    expect(parsePreviewDesignCommand({ extra: true, session: 1, type: 'start' })).toBeNull()
    expect(parsePreviewDesignCommand({ session: 0, type: 'start' })).toBeNull()
    expect(parsePreviewDesignBridgeEvent({ session: 2, type: 'ready' }, 1)).toBeNull()
    expect(parsePreviewDesignBridgeEvent({ session: 1, type: 'ready' }, 1)).toEqual({ session: 1, type: 'ready' })
  })
})

describe('Design Mode DOM Layers', () => {
  it('has no visible mark before selection and neutralizes pointer, touch, mouse, and keyboard activation', async () => {
    const frame = document.createElement('iframe')
    document.body.appendChild(frame)

    try {
      const doc = frame.contentDocument!
      const button = doc.createElement('button')
      doc.body.appendChild(button)
      doc.elementFromPoint = () => button
      const guest = designInPage(doc)
      const activated: string[] = []

      for (const type of ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'click', 'touchstart', 'touchend']) {
        button.addEventListener(type, () => activated.push(type))
      }

      button.addEventListener('keydown', event => activated.push(`key:${event.key}`))
      guest.install()
      const outline = doc.querySelector('hermes-design')!.shadowRoot!.querySelector<HTMLElement>('[data-design-selected]')!
      expect(outline.style.display).toBe('none')

      for (const type of ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'touchstart', 'touchend']) {
        const event = new Event(type, { bubbles: true, cancelable: true })
        expect(button.dispatchEvent(event)).toBe(false)
      }

      const click = new MouseEvent('click', { bubbles: true, button: 0, cancelable: true })
      expect(button.dispatchEvent(click)).toBe(false)
      await expect(guest.wait()).resolves.toMatchObject({ type: 'select' })
      expect(outline.style.display).toBe('block')

      button.focus()

      for (const key of ['Enter', ' ', 'Spacebar']) {
        const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key })
        expect(button.dispatchEvent(event)).toBe(false)
      }

      expect(activated).toEqual([])
      guest.teardown()
    } finally {
      frame.remove()
    }
  })

  it('lets Escape end selection without activating the page', async () => {
    const frame = document.createElement('iframe')
    document.body.appendChild(frame)

    try {
      const doc = frame.contentDocument!
      const button = doc.createElement('button')
      doc.body.appendChild(button)
      const guest = designInPage(doc)
      guest.install()
      button.focus()
      const escape = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Escape' })
      button.dispatchEvent(escape)
      await expect(guest.wait()).resolves.toEqual({ type: 'end' })
      expect(escape.defaultPrevented).toBe(true)
      guest.teardown()
    } finally {
      frame.remove()
    }
  })

  it('selects an ancestor and returns to the original child without activating the page', async () => {
    const frame = document.createElement('iframe')
    document.body.appendChild(frame)

    try {
      const doc = frame.contentDocument!
      const section = doc.createElement('section')
      const button = doc.createElement('button')
      section.appendChild(button)
      doc.body.appendChild(section)
      doc.elementFromPoint = () => button
      const guest = designInPage(doc)
      guest.install()
      frame.contentWindow!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, button: 0 }))
      const first = await guest.wait()
      expect(first.type).toBe('select')

      if (first.type !== 'select') {
        throw new Error('Expected a selection')
      }

      expect(first.selection.layers.map(layer => layer.tag)).toEqual(['body', 'section', 'button'])
      expect(first.selection.selectedLayer).toBe(2)
      expect(guest.selectLayer(1)).toBe(true)
      const parent = await guest.wait()
      expect(parent.type).toBe('select')

      if (parent.type !== 'select') {
        throw new Error('Expected a parent selection')
      }

      expect(parent.selection.tag).toBe('section')
      expect(parent.selection.selectedLayer).toBe(1)
      expect(guest.selectLayer(2)).toBe(true)
      const child = await guest.wait()
      expect(child.type).toBe('select')

      if (child.type !== 'select') {
        throw new Error('Expected a child selection')
      }

      expect(child.selection.tag).toBe('button')
      expect(guest.selectLayer(50)).toBe(false)
      guest.teardown()
    } finally {
      frame.remove()
    }
  })
})

describe('Design Mode guest teardown', () => {
  it('delivers end when teardown happens before the host begins waiting', async () => {
    const frame = document.createElement('iframe')
    document.body.appendChild(frame)

    try {
      const guest = designInPage(frame.contentDocument!)
      guest.install()
      guest.teardown()
      await expect(guest.wait()).resolves.toEqual({ type: 'end' })
    } finally {
      frame.remove()
    }
  })
})
