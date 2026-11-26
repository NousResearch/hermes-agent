import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { group, split } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $hiddenTreePanes, $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $approvalModes } from '@/store/approval-mode'
import { $focusedTreePaneId } from '@/store/session-focus'
import { $statusbarHiddenIds, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { useApprovalModeStatusbarItem } from './approval-mode-menu'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  $approvalModes.set({})
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
})

function Harness({
  gatewayState = 'open',
  profile = 'default',
  requestGateway,
  sessionYolo
}: {
  gatewayState?: string
  profile?: string
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
  sessionYolo?: { active: boolean; onToggle: (enabled: boolean) => void }
}) {
  const item = useApprovalModeStatusbarItem(profile, requestGateway, gatewayState, sessionYolo)

  // The bar spreads this item with a `toggleLabel` (use-statusbar-items), which
  // is what makes it hideable via the context menu — mirror that shape.
  return (
    <MemoryRouter>
      <StatusbarControls items={[{ ...item, toggleLabel: 'Approval mode' }]} />
    </MemoryRouter>
  )
}

describe('approval mode statusbar item', () => {
  it('lights the zap and says YOLO when the session bypass is on but the profile mode is not off', async () => {
    $statusbarHiddenIds.set([])
    const requestGateway = vi.fn(async () => ({ value: 'smart' }))
    const onToggle = vi.fn()
    const { rerender } = render(<Harness requestGateway={requestGateway} sessionYolo={{ active: false, onToggle }} />)

    const before = await screen.findByRole('button', { name: /smart/i })
    expect(before.textContent).not.toMatch(/yolo/i)

    rerender(<Harness requestGateway={requestGateway} sessionYolo={{ active: true, onToggle }} />)

    const trigger = screen.getByRole('button', { name: /yolo/i })
    expect(trigger.textContent).not.toMatch(/smart/i)
    expect(trigger.className).toContain('bg-(--chrome-action-hover)')

    fireEvent.pointerDown(trigger, { button: 0 })
    const row = await screen.findByRole('menuitemcheckbox', { name: /yolo for this chat/i })
    expect(row.getAttribute('aria-checked')).toBe('true')
    expect(row.getAttribute('aria-disabled')).not.toBe('true')

    fireEvent.click(row)
    expect(onToggle).toHaveBeenCalledWith(false)
  })

  it('keeps the profile mode label when it is off and disables the redundant session row', async () => {
    $statusbarHiddenIds.set(['approval-mode'])
    const requestGateway = vi.fn(async () => ({ value: 'off' }))
    render(<Harness requestGateway={requestGateway} sessionYolo={{ active: true, onToggle: vi.fn() }} />)

    const trigger = await screen.findByRole('button', { name: /^off$/i })
    expect(trigger.textContent).not.toMatch(/yolo/i)

    fireEvent.pointerDown(trigger, { button: 0 })
    const row = await screen.findByRole('menuitemcheckbox', { name: /yolo for this chat/i })
    expect(row.getAttribute('aria-disabled')).toBe('true')
  })

  it('renders no session row when the surface has no session yolo control', async () => {
    $statusbarHiddenIds.set([])
    const response = new Promise<never>(() => undefined)
    render(<Harness requestGateway={vi.fn(() => response)} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: /smart/i }), { button: 0 })
    await screen.findByRole('menuitemradio', { name: /manual/i })
    expect(screen.queryByRole('menuitemcheckbox')).toBeNull()
  })
  it('surfaces a user-hidden pill while a bypass is active, and hides it again after', async () => {
    $statusbarHiddenIds.set(['approval-mode'])
    const requestGateway = vi.fn(async () => ({ value: 'smart' }))
    const onToggle = vi.fn()
    const { rerender } = render(<Harness requestGateway={requestGateway} sessionYolo={{ active: false, onToggle }} />)

    await waitFor(() => expect(requestGateway).toHaveBeenCalled())
    expect(screen.queryByRole('button', { name: /smart/i })).toBeNull()

    rerender(<Harness requestGateway={requestGateway} sessionYolo={{ active: true, onToggle }} />)
    expect(screen.getByRole('button', { name: /yolo/i })).toBeTruthy()

    rerender(<Harness requestGateway={requestGateway} sessionYolo={{ active: false, onToggle }} />)
    expect(screen.queryByRole('button', { name: /yolo|smart/i })).toBeNull()
  })

  it('scopes the hint to workspaces that declare per-profile approvals without extra mode reads', async () => {
    const previousTree = $layoutTree.get()
    const previousGroup = $activeTreeGroup.get()
    const previousHidden = $hiddenTreePanes.get()
    const room = 'plugin-workspace:room'
    const generic = 'plugin-workspace:generic'
    const requestGateway = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ value: 'off' }))
    const configReads = () => requestGateway.mock.calls.filter(([method]) => method === 'config.get').length

    const disposers = [
      registry.register({ area: 'panes', data: { approvalScope: 'per-profile' }, id: room }),
      registry.register({ area: 'panes', data: {}, id: generic })
    ]

    const layout = (roomActive: string, genericGroup = false) =>
      split('row', [
        group(['workspace'], { id: 'chat' }),
        group(genericGroup ? [generic] : [room, 'preview'], {
          active: genericGroup ? generic : roomActive,
          id: 'plugin'
        }),
        group(['files'], { id: 'side' })
      ])

    try {
      act(() => {
        $layoutTree.set(layout(room))
        $activeTreeGroup.set('chat')
        $activeTreeGroup.set('plugin')
      })
      render(<Harness requestGateway={requestGateway} />)
      expect(screen.getByRole('button', { name: 'Per profile' })).toBeTruthy()
      // Navigation never re-reads the mode: a read issued mid-write would supersede the write.
      expect(configReads()).toBe(1)

      act(() => $activeTreeGroup.set('side'))
      expect(screen.getByRole('button', { name: 'Per profile' })).toBeTruthy()
      expect($focusedTreePaneId.get()).toBe('workspace')

      // A preview fronting the room's group covers the room without leaving it.
      act(() => {
        $layoutTree.set(layout('preview'))
        $activeTreeGroup.set('plugin')
      })
      expect(screen.getByRole('button', { name: 'Per profile' })).toBeTruthy()
      act(() => $hiddenTreePanes.set(new Set([room])))
      await waitFor(() => expect(screen.getByRole('button', { name: /^off$/i })).toBeTruthy())
      act(() => $hiddenTreePanes.set(new Set()))

      act(() => $activeTreeGroup.set('chat'))
      await waitFor(() => expect(screen.getByRole('button', { name: /^off$/i })).toBeTruthy())

      act(() => {
        $layoutTree.set(layout(room))
        $activeTreeGroup.set('plugin')
      })
      expect(screen.getByRole('button', { name: 'Per profile' })).toBeTruthy()
      act(() => $hiddenTreePanes.set(new Set([room])))
      expect(screen.getByRole('button', { name: /^off$/i })).toBeTruthy()
      act(() => $hiddenTreePanes.set(new Set()))

      // A workspace that declares no approval scope keeps the profile's menu.
      act(() => {
        $layoutTree.set(layout(room, true))
        $activeTreeGroup.set('plugin')
      })
      expect(screen.getByRole('button', { name: /^off$/i })).toBeTruthy()

      // A preview covering the remembered chat keeps the chat's menu while a room is visible beside it.
      const chatBehindPreview = (active: string) =>
        split('row', [group(['workspace', 'preview'], { active, id: 'chat' }), group([room], { id: 'plugin' })])

      act(() => {
        $layoutTree.set(chatBehindPreview('workspace'))
        $activeTreeGroup.set('chat')
      })
      act(() => $layoutTree.set(chatBehindPreview('preview')))
      expect(screen.getByRole('button', { name: /^off$/i })).toBeTruthy()
      expect(configReads()).toBe(1)
    } finally {
      cleanup()
      disposers.forEach(dispose => dispose())
      $hiddenTreePanes.set(previousHidden)
      $layoutTree.set(previousTree)
      $activeTreeGroup.set(previousGroup)
    }
  })

  it('writes the selected mode through the gateway and updates its shared trigger label', async () => {
    $statusbarHiddenIds.set([])
    const requestGateway = vi.fn(async (_method, params) => ({ value: params?.value ?? 'smart' }))
    render(<Harness profile="work" requestGateway={requestGateway} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: /smart/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /manual/i }))

    await waitFor(() => {
      // The menu shows the "work" profile, so the write must name it (#125969); an unscoped
      // `config.set` would edit whichever profile the backend was launched with instead.
      expect(requestGateway).toHaveBeenCalledWith('config.set', {
        key: 'approvals.mode',
        value: 'manual',
        profile: 'work'
      })
      expect(screen.getByRole('button', { name: /manual/i })).toBeTruthy()
    })
  })

  it('syncs once the gateway opens instead of sitting on the pre-sync default', async () => {
    // Boot order: the shell (and this chip) mount while the socket is still
    // connecting, so the sync must wait for 'open' — and then actually run.
    // A sync attempted while connecting rejects ("Hermes gateway
    // unavailable") and, swallowed, would leave the chip on its optimistic
    // 'smart' default for the whole session even though the configured mode
    // is 'off'.
    const requestGateway = vi.fn(async () => ({ value: 'off' }))

    const { rerender } = render(<Harness gatewayState="connecting" profile="boot" requestGateway={requestGateway} />)

    expect(requestGateway).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /smart/i })).toBeTruthy()

    rerender(<Harness gatewayState="open" profile="boot" requestGateway={requestGateway} />)

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.get', { key: 'approvals.mode', profile: 'boot' })
      expect(screen.getByRole('button', { name: /off/i })).toBeTruthy()
    })
  })
})
