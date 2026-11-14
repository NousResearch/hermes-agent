import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { group } from '@/components/pane-shell/tree/model'
import { TreeGroup } from '@/components/pane-shell/tree/renderer/tree-group'
import { provideZoneHeader } from '@/components/pane-shell/tree/renderer/zone-header-provider'
import { registry } from '@/contrib/registry'
import { $shellChoice } from '@/store/shell-variant'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { usePortalZoneHeader } from './portal-conversation-header'

// The contrib controller installs this at boot; TreeGroup alone draws only the shared strip.
beforeAll(() => provideZoneHeader(usePortalZoneHeader))

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

it('replaces chat tabs without remounting drafts, preserves mixed tabs and minimized restore handles', () => {
  stubMenuDomApis()
  stubResizeObserver()
  $workspaceIsPage.set(false)
  const mounted = vi.fn()
  const unmounted = vi.fn()

  function Chat() {
    useEffect(() => {
      mounted()

      return unmounted
    }, [])

    return <input aria-label="Draft" defaultValue="" />
  }

  function Navigation() {
    const navigate = useNavigate()

    return (
      <>
        <button onClick={() => navigate('/settings')}>Open settings page</button>
        <button onClick={() => navigate('/')}>Return to chat</button>
      </>
    )
  }

  const dispose = registry.register({
    area: 'panes',
    id: 'workspace',
    title: 'My chat',
    data: { placement: 'main' },
    render: Chat
  })

  const disposeTool = registry.register({
    area: 'panes',
    id: 'test-tool',
    title: 'Tool',
    data: { placement: 'main' },
    render: () => <p>Tool</p>
  })

  const node = group(['workspace'], { id: 'main' })

  const host = (value = node) => (
    <MemoryRouter>
      <Navigation />
      <TreeGroup node={value} parentAxis="column" topEdge={false} />
    </MemoryRouter>
  )

  // The main chat's stage header (its Agent settings rides the band).
  const stageHeader = () => document.querySelector('[data-slot="portal-conversation-header"]')

  try {
    const view = render(host())
    expect(stageHeader()).toBeTruthy()
    expect(screen.queryByRole('tab', { name: /My chat/ })).toBeNull()
    fireEvent.change(screen.getByRole('textbox', { name: 'Draft' }), { target: { value: 'Keep this draft' } })
    fireEvent.click(screen.getByRole('button', { name: 'Open settings page' }))
    expect(stageHeader()).toBeNull()
    expect((screen.getByRole('textbox', { name: 'Draft' }) as HTMLInputElement).value).toBe('Keep this draft')
    fireEvent.click(screen.getByRole('button', { name: 'Return to chat' }))
    expect(stageHeader()).toBeTruthy()
    act(() => $shellChoice.set(false))
    view.rerender(host())
    expect(screen.getByRole('tab', { name: /My chat/ })).toBeTruthy()
    expect((screen.getByRole('textbox', { name: 'Draft' }) as HTMLInputElement).value).toBe('Keep this draft')
    expect(mounted).toHaveBeenCalledTimes(1)
    expect(unmounted).not.toHaveBeenCalled()
    act(() => $shellChoice.set(true))
    view.rerender(host({ ...node, panes: ['workspace', 'test-tool'] }))
    expect(screen.getByRole('tab', { name: /Tool/ })).toBeTruthy()
    expect(stageHeader()).toBeTruthy()
    view.rerender(host({ ...node, minimized: true }))
    expect(stageHeader()).toBeNull()
    expect(screen.getByRole('tab', { name: /My chat/ })).toBeTruthy()
  } finally {
    cleanup()
    act(() => {
      dispose()
      disposeTool()
    })
  }
})

it('lists every open chat once a second chat tab joins the main chat (the band strip), not just one title', () => {
  stubMenuDomApis()
  stubResizeObserver()
  $workspaceIsPage.set(false)

  const disposers = [
    registry.register({ area: 'panes', id: 'workspace', title: 'Funnel review', data: { placement: 'main' }, render: () => null }),
    registry.register({
      area: 'panes',
      id: 'session-tile:a',
      title: 'Hiring loop',
      data: { placement: 'main' },
      render: () => null
    })
  ]

  try {
    render(
      <MemoryRouter>
        <TreeGroup
          node={group(['workspace', 'session-tile:a'], { id: 'main', active: 'session-tile:a' })}
          parentAxis="column"
          topEdge={false}
        />
      </MemoryRouter>
    )
    expect(screen.getByRole('tab', { name: /Funnel review/ })).toBeTruthy()
    expect(screen.getByRole('tab', { name: /Hiring loop/, selected: true })).toBeTruthy()
    // The conversation chrome stays beside the strip.
    expect(screen.getByRole('button', { name: 'Agent settings' })).toBeTruthy()
  } finally {
    cleanup()
    act(() => disposers.forEach(dispose => dispose()))
  }
})
