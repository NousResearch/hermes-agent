import { afterEach, describe, expect, it, vi } from 'vitest'

const openCanvasWindow = vi.fn(async () => ({ ok: true }))
const closeCanvasWindow = vi.fn(async () => {})
let popoutClosed: ((provider: string) => void) | null = null

Object.assign(window, {
  hermesDesktop: {
    openCanvasWindow,
    closeCanvasWindow,
    onCanvasPopoutClosed: (cb: (provider: string) => void) => {
      popoutClosed = cb

      return () => {}
    },
    onCanvasPopoutTab: () => () => {}
  }
})

import {
  $canvasTabs,
  $dockedCanvasTabs,
  canvasTileOpen,
  closeCanvasTile,
  dismissCanvasTile,
  openCanvasTile,
  popOutCanvasTile,
  registerCanvasProvider,
  watchCanvasTiles
} from './canvas-tile'

const tab = (docId: string) => ({ docId, provider: 'pen', title: docId.toUpperCase(), url: 'https://app.pen.dev/new?embed' })
const popOut = vi.fn()

registerCanvasProvider({
  id: 'pen',
  untitled: 'Canvas',
  tabLead: () => null,
  render: () => null,
  close: () => {},
  popOut
})
watchCanvasTiles()

afterEach(() => {
  dismissCanvasTile('pen')
  vi.clearAllMocks()
})

describe('openCanvasTile', () => {
  it('keeps one pane per provider when the live .pen changes', () => {
    openCanvasTile(tab('a'))
    openCanvasTile(tab('b'))

    expect($canvasTabs.get()).toEqual([tab('b')])
  })
})

describe('popped-out canvas', () => {
  it('leaves the docked tree, hands the provider its guest, and comes back when the window closes', async () => {
    openCanvasTile(tab('a'))
    popOutCanvasTile('pen')

    expect(popOut).toHaveBeenCalledTimes(1)
    expect(openCanvasWindow).toHaveBeenCalledWith(tab('a'))
    expect($dockedCanvasTabs.get()).toEqual([])
    // Still "open" for anyone asking — it is just elsewhere.
    expect(canvasTileOpen('pen')).toBe(true)

    await Promise.resolve()
    popoutClosed?.('pen')

    expect($dockedCanvasTabs.get()).toEqual([tab('a')])
  })

  it('forwards a new document to the window instead of seating a docked tile', () => {
    openCanvasTile(tab('a'))
    popOutCanvasTile('pen')
    openCanvasTile(tab('b'))

    expect($dockedCanvasTabs.get()).toEqual([])
    expect(openCanvasWindow).toHaveBeenLastCalledWith(tab('b'))

    popoutClosed?.('pen')

    // The tile that comes back is the document the window was last showing.
    expect($dockedCanvasTabs.get()).toEqual([tab('b')])
  })

  it('takes the window down with the canvas when the document is gone', () => {
    openCanvasTile(tab('a'))
    popOutCanvasTile('pen')
    dismissCanvasTile('pen')

    expect(closeCanvasWindow).toHaveBeenCalledWith('pen')
    expect(canvasTileOpen('pen')).toBe(false)

    // The window's own closed signal must not resurrect the tile.
    popoutClosed?.('pen')
    expect($dockedCanvasTabs.get()).toEqual([])
  })

  it('seats the tile back when the window fails to open', async () => {
    openCanvasWindow.mockResolvedValueOnce({ ok: false })
    openCanvasTile(tab('a'))
    popOutCanvasTile('pen')

    await vi.waitFor(() => expect($dockedCanvasTabs.get()).toEqual([tab('a')]))
    closeCanvasTile('pen')
  })
})
