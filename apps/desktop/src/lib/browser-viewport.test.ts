import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { installBrowserViewport } from './browser-viewport'

let dispose: (() => void) | undefined
let root: HTMLDivElement
let viewport: EventTarget & { height: number; offsetTop: number; scale: number }
let frames: Map<number, FrameRequestCallback>
let nextFrame: number

const flush = () => {
  const pending = [...frames.values()]
  frames.clear()
  pending.forEach(callback => callback(0))
}

const resize = (height: number, offsetTop = 0, scale = 1) => {
  Object.assign(viewport, { height, offsetTop, scale })
  viewport.dispatchEvent(new Event('resize'))
  viewport.dispatchEvent(new Event('scroll'))
  flush()
}

beforeEach(() => {
  root = document.createElement('div')
  root.id = 'root'
  document.body.append(root)
  document.documentElement.dataset.hermesDesktopHost = 'browser'
  viewport = Object.assign(new EventTarget(), { height: 800, offsetTop: 0, scale: 1 })
  frames = new Map()
  nextFrame = 0
  vi.stubGlobal('innerHeight', 800)
  vi.stubGlobal('visualViewport', viewport)
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const id = ++nextFrame
    frames.set(id, callback)

    return id
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
})

afterEach(() => {
  dispose?.()
  dispose = undefined
  root.remove()
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  document.documentElement.style.removeProperty('zoom')
  vi.unstubAllGlobals()
})

describe('browser visible viewport', () => {
  it('fits a keyboard and its pan at UI scale, without changing focus, pinch layout, or closed-keyboard layout', () => {
    const editor = document.createElement('textarea')
    editor.value = 'first line\nsecond line'
    root.append(editor)
    editor.focus()
    dispose = installBrowserViewport()
    expect(root.hasAttribute('data-browser-viewport')).toBe(false)

    resize(460, 25)
    expect(root.style.getPropertyValue('--browser-viewport-height')).toBe('460px')
    expect(root.style.getPropertyValue('--browser-viewport-top')).toBe('25px')
    expect(document.activeElement).toBe(editor)
    expect(editor.value).toBe('first line\nsecond line')

    resize(230, 90, 2)
    expect(root.style.getPropertyValue('--browser-viewport-height')).toBe('460px')
    expect(root.style.getPropertyValue('--browser-viewport-top')).toBe('25px')
    // Closing/reopening the keyboard while still pinched changes available
    // space, not the user's zoom. It must not leave the old keyboard inset.
    resize(400, 90, 2)
    expect(root.hasAttribute('data-browser-viewport')).toBe(false)
    resize(230, 90, 2)
    expect(root.style.getPropertyValue('--browser-viewport-height')).toBe('460px')
    expect(root.style.getPropertyValue('--browser-viewport-top')).toBe('0px')

    document.documentElement.style.zoom = '1.25'
    resize(460, 25)
    expect(root.style.getPropertyValue('--browser-viewport-height')).toBe('368px')
    expect(root.style.getPropertyValue('--browser-viewport-top')).toBe('20px')
    editor.blur()
    resize(500, 0)
    expect(document.activeElement).not.toBe(editor)
    expect(root.hasAttribute('data-browser-viewport')).toBe(true)

    // Rotation/layout-resizing browsers already made room for the keyboard.
    vi.stubGlobal('innerHeight', 500)
    window.dispatchEvent(new Event('resize'))
    flush()
    expect(root.hasAttribute('data-browser-viewport')).toBe(false)
    expect(root.style.getPropertyValue('--browser-viewport-height')).toBe('')
    vi.stubGlobal('innerHeight', 800)
    resize(800)
    expect(root.hasAttribute('data-browser-viewport')).toBe(false)
    expect(editor.value).toBe('first line\nsecond line')
  })

  it('leaves native/missing-API hosts alone and releases coalesced listeners and layout on teardown', () => {
    document.documentElement.removeAttribute('data-hermes-desktop-host')
    installBrowserViewport()()
    resize(400)
    expect(root.hasAttribute('data-browser-viewport')).toBe(false)
    expect(frames.size).toBe(0)

    document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.stubGlobal('visualViewport', null)
    installBrowserViewport()()
    expect(root.hasAttribute('data-browser-viewport')).toBe(false)
    vi.stubGlobal('visualViewport', viewport)
    dispose = installBrowserViewport()
    expect(root.hasAttribute('data-browser-viewport')).toBe(true)
    viewport.dispatchEvent(new Event('resize'))
    viewport.dispatchEvent(new Event('scroll'))
    window.dispatchEvent(new Event('resize'))
    expect(frames.size).toBe(1)
    dispose()
    expect(frames.size).toBe(0)
    resize(300)
    window.dispatchEvent(new Event('resize'))
    flush()
    expect(root.hasAttribute('data-browser-viewport')).toBe(false)
    expect(root.style.getPropertyValue('--browser-viewport-top')).toBe('')
    expect(frames.size).toBe(0)
  })
})
