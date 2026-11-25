import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from './context-menu'
import { Dialog, DialogContent, DialogTitle } from './dialog'
import { type BrowserPortalLayer, getBrowserPortalContainer, usePopoverPortalContainer } from './dialog-portal-context'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './select'
import { RootTooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from './tooltip'

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
})

afterEach(() => {
  cleanup()
  delete window.document.documentElement.dataset.hermesDesktopHost

  for (const host of window.document.querySelectorAll('body > [data-browser-portal-host]')) {
    host.remove()
  }
})

function Probe({ explicit, layer }: { explicit?: HTMLElement | null; layer?: BrowserPortalLayer }) {
  const container = usePopoverPortalContainer(explicit, layer)

  return <span data-host={container?.dataset.browserPortalHost ?? container?.dataset.slot ?? 'body'} />
}

describe('browser portal hosts', () => {
  it('returns undefined and creates no hosts on native Desktop', () => {
    expect(getBrowserPortalContainer('menu')).toBeUndefined()
    expect(getBrowserPortalContainer('select')).toBeUndefined()
    expect(getBrowserPortalContainer('tooltip')).toBeUndefined()
    expect(window.document.querySelectorAll('body > [data-browser-portal-host]').length).toBe(0)
  })

  it('creates and reuses one body-level host per layer on the Webapp', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'

    const menuA = getBrowserPortalContainer('menu')
    const menuB = getBrowserPortalContainer('menu')
    const select = getBrowserPortalContainer('select')
    const tooltip = getBrowserPortalContainer('tooltip')

    expect(menuA).toBeDefined()
    expect(menuB).toBe(menuA)
    expect(menuA?.parentElement).toBe(window.document.body)
    expect(select).not.toBe(menuA)
    expect(tooltip).not.toBe(menuA)
    expect(tooltip).not.toBe(select)
    expect(window.document.querySelectorAll('body > [data-browser-portal-host]').length).toBe(3)

    // Re-creates cleanly if removed between mounts.
    menuA?.remove()
    const menuC = getBrowserPortalContainer('menu')
    expect(menuC).toBeDefined()
    expect(menuC).not.toBe(menuA)
    expect(menuC?.parentElement).toBe(window.document.body)
  })

  it('keeps dialog and explicit containers ahead of the browser fallback', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    const explicit = window.document.createElement('div')
    explicit.dataset.slot = 'explicit-host'

    const { rerender } = render(<Probe layer="menu" />)
    expect(window.document.querySelector('span[data-host]')?.getAttribute('data-host')).toBe('menu')

    rerender(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Dialog</DialogTitle>
          <Probe layer="select" />
        </DialogContent>
      </Dialog>
    )
    expect(window.document.querySelector('span[data-host]')?.getAttribute('data-host')).toBe('dialog-content')

    rerender(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Dialog</DialogTitle>
          <Probe explicit={explicit} layer="menu" />
        </DialogContent>
      </Dialog>
    )
    expect(window.document.querySelector('span[data-host]')?.getAttribute('data-host')).toBe('explicit-host')
  })

  it('routes ContextMenu, Select, and Tooltip into their browser hosts on the Webapp', () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'

    const first = render(
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <button type="button">Context target</button>
        </ContextMenuTrigger>
        <ContextMenuContent>
          <ContextMenuItem>Context action</ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'Context target' }))
    expect(
      screen.getByRole('menuitem', { name: 'Context action' }).closest('[data-browser-portal-host="menu"]')
    ).not.toBeNull()
    first.unmount()

    const second = render(
      <Select defaultOpen defaultValue="a">
        <SelectTrigger aria-label="Picker">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a">Alpha</SelectItem>
        </SelectContent>
      </Select>
    )

    expect(screen.getByRole('option', { name: 'Alpha' }).closest('[data-browser-portal-host="select"]')).not.toBeNull()
    second.unmount()

    render(
      <RootTooltipProvider>
        <Tooltip open>
          <TooltipTrigger asChild>
            <button type="button">Tip target</button>
          </TooltipTrigger>
          <TooltipContent>Tip body</TooltipContent>
        </Tooltip>
      </RootTooltipProvider>
    )
    expect(
      window.document.querySelector('[data-slot="tooltip-content"]')?.closest('[data-browser-portal-host="tooltip"]')
    ).not.toBeNull()
  })
})
