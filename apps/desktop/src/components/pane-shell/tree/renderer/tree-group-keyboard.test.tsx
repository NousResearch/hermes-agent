import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, split } from '../model'
import { $layoutTree, declareDefaultTree } from '../store'

import { LayoutTreeRoot } from '.'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    callback(performance.now())

    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', vi.fn())

  for (const id of ['workspace', 'preview-tile:alpha', 'preview-tile:beta', 'preview-tile:gamma']) {
    disposers.push(
      registry.register({
        area: 'panes',
        data: id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
        id,
        render: () => <div>{id}</div>,
        title: id.replace('preview-tile:', '')
      })
    )
  }
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $layoutTree.set(null)
})

describe('pane tab keyboard semantics', () => {
  it('uses roving focus and ARIA tab/panel wiring for horizontal preview tile tabs', () => {
    declareDefaultTree(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        group(['preview-tile:alpha', 'preview-tile:beta', 'preview-tile:gamma'], {
          active: 'preview-tile:alpha',
          id: 'grp-preview'
        })
      ])
    )

    render(<LayoutTreeRoot />)

    const tabs = screen.getAllByRole('tab').filter(tab => ['alpha', 'beta', 'gamma'].includes(tab.textContent ?? ''))

    expect(tabs.map(tab => tab.textContent)).toEqual(['alpha', 'beta', 'gamma'])
    expect(tabs[0].getAttribute('aria-selected')).toBe('true')
    expect(tabs[0].getAttribute('tabindex')).toBe('0')
    expect(tabs[1].getAttribute('tabindex')).toBe('-1')

    const panelId = tabs[0].getAttribute('aria-controls')
    expect(panelId).toBeTruthy()
    expect(document.getElementById(panelId!)?.getAttribute('role')).toBe('tabpanel')
    expect(document.getElementById(panelId!)?.getAttribute('aria-labelledby')).toBe(tabs[0].id)

    fireEvent.keyDown(tabs[0], { key: 'ArrowRight' })
    expect(screen.getByRole('tab', { name: 'beta' }).getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: 'beta' }))

    fireEvent.keyDown(screen.getByRole('tab', { name: 'beta' }), { key: 'End' })
    expect(screen.getByRole('tab', { name: 'gamma' }).getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: 'gamma' }))

    fireEvent.keyDown(screen.getByRole('tab', { name: 'gamma' }), { key: 'ArrowRight' })
    expect(screen.getByRole('tab', { name: 'alpha' }).getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: 'alpha' }))
  })
})
