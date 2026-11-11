import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { group } from '@/components/pane-shell/tree/model'
import { TreeGroup } from '@/components/pane-shell/tree/renderer/tree-group'
import { registry } from '@/contrib/registry'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { usePortalRedesign } from './use-portal-redesign'

vi.mock('./use-portal-redesign', () => ({ usePortalRedesign: vi.fn(() => true) }))

afterEach(cleanup)

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

  try {
    const view = render(host())
    expect(screen.getByRole('heading', { name: 'My chat' })).toBeTruthy()
    expect(screen.queryByRole('tab', { name: /My chat/ })).toBeNull()
    fireEvent.change(screen.getByRole('textbox', { name: 'Draft' }), { target: { value: 'Keep this draft' } })
    fireEvent.click(screen.getByRole('button', { name: 'Open settings page' }))
    expect(screen.queryByRole('heading', { name: 'My chat' })).toBeNull()
    expect((screen.getByRole('textbox', { name: 'Draft' }) as HTMLInputElement).value).toBe('Keep this draft')
    fireEvent.click(screen.getByRole('button', { name: 'Return to chat' }))
    expect(screen.getByRole('heading', { name: 'My chat' })).toBeTruthy()
    vi.mocked(usePortalRedesign).mockReturnValue(false)
    view.rerender(host())
    expect(screen.getByRole('tab', { name: /My chat/ })).toBeTruthy()
    expect((screen.getByRole('textbox', { name: 'Draft' }) as HTMLInputElement).value).toBe('Keep this draft')
    expect(mounted).toHaveBeenCalledTimes(1)
    expect(unmounted).not.toHaveBeenCalled()
    vi.mocked(usePortalRedesign).mockReturnValue(true)
    view.rerender(host({ ...node, panes: ['workspace', 'test-tool'] }))
    expect(screen.getByRole('tab', { name: /Tool/ })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'My chat' })).toBeTruthy()
    view.rerender(host({ ...node, minimized: true }))
    expect(screen.queryByRole('heading', { name: 'My chat' })).toBeNull()
    expect(screen.getByRole('tab', { name: /My chat/ })).toBeTruthy()
  } finally {
    cleanup()
    act(() => {
      dispose()
      disposeTool()
    })
  }
})
