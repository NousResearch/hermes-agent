import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { StableNavigateProvider } from '@/app/stable-navigate'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { en } from '@/i18n/en'
import { $confirmRequest } from '@/store/confirm'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $fleetRoster } from '@/store/fleet-roster'
import { $interfaceMode } from '@/store/interface-mode'
import { $gatewayState } from '@/store/session'
import { stubMenuDomApis } from '@/test/jsdom'

const mocks = vi.hoisted(() => ({ restart: vi.fn(), reconnect: vi.fn(() => Promise.resolve()) }))

vi.mock('@/store/system-actions', () => ({ runGatewayRestart: mocks.restart }))
vi.mock('@/store/gateway-reconnect', () => ({ reconnectGateway: mocks.reconnect }))
// The panel's live status read is the statusbar's own hook; here the machine is simply up.
vi.mock('@/app/shell/hooks/use-status-snapshot', () => ({
  useStatusSnapshot: () => ({
    inferenceStatus: { ready: true, source: 'probe' },
    statusSnapshot: { gateway_platforms: { telegram: { state: 'connected' } } }
  })
}))

import { SimplifiedGatewayHeader } from './simplified-gateway-header'

const copy = en.shell.gatewayMenu
const onToggle = vi.fn()

function mount(props: { connectionId?: null | string; label?: string; selected?: boolean } = {}) {
  return render(
    <MemoryRouter>
      <StableNavigateProvider>
        <RootTooltipProvider>
          <SimplifiedGatewayHeader
            connectionId={props.connectionId ?? null}
            label={props.label ?? 'Hermes host'}
            onToggle={onToggle}
            open
            selected={props.selected ?? true}
          />
        </RootTooltipProvider>
      </StableNavigateProvider>
    </MemoryRouter>
  )
}

const statusButton = (label = 'Hermes host', status = copy.connected) =>
  screen.getByRole('button', { name: copy.machineStatus(label, status) })

const popover = () => screen.queryByRole('dialog')

beforeEach(() => {
  stubMenuDomApis()
  onToggle.mockReset()
  mocks.restart.mockReset()
  $gatewayState.set('open')
  $interfaceMode.set('simple')
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  $confirmRequest.set(null)
  $fleetRoster.set(null)
  $connectionsRegistry.set(null)
  $gatewayState.set('idle')
  $interfaceMode.set('advanced')
})

describe('a machine row status', () => {
  it('opens the machine popover by click without toggling the machine, and Escape returns focus', async () => {
    mount()
    const status = statusButton()
    expect(status.getAttribute('aria-haspopup')).toBe('dialog')

    fireEvent.click(status)
    expect(popover()).not.toBeNull()
    expect(onToggle).not.toHaveBeenCalled()
    expect(screen.getByText(copy.inferenceReady)).toBeTruthy()

    fireEvent.keyDown(popover()!, { key: 'Escape' })
    // The focus scope hands focus back on the task after it unmounts.
    await act(() => new Promise(resolve => setTimeout(resolve, 0)))
    expect(popover()).toBeNull()
    expect(window.document.activeElement).toBe(status)
  })

  it('opens on a resting pointer after a delay, without taking focus, and closes once it leaves', () => {
    vi.useFakeTimers()
    mount()
    const status = statusButton()

    fireEvent.pointerEnter(status, { pointerType: 'mouse' })
    act(() => vi.advanceTimersByTime(250))
    expect(popover()).toBeNull()
    act(() => vi.advanceTimersByTime(100))
    expect(popover()).not.toBeNull()
    expect(popover()!.contains(window.document.activeElement)).toBe(false)

    // Travelling from the status into the popover keeps it open.
    fireEvent.pointerLeave(status, { pointerType: 'mouse' })
    fireEvent.pointerEnter(popover()!, { pointerType: 'mouse' })
    act(() => vi.advanceTimersByTime(500))
    expect(popover()).not.toBeNull()

    fireEvent.pointerLeave(popover()!, { pointerType: 'mouse' })
    act(() => vi.advanceTimersByTime(250))
    expect(popover()).toBeNull()
  })

  it('shows Simple: reconnect only, platforms, and recent activity folded', () => {
    mount()
    fireEvent.click(statusButton())
    expect(screen.getByRole('button', { name: copy.reconnectGateway })).toBeTruthy()
    expect(screen.queryByRole('button', { name: en.commandCenter.restartGateway })).toBeNull()
    expect(screen.getByText('telegram')).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.recentActivity }).getAttribute('aria-expanded')).toBe('false')
  })

  it('adds the system panel and a confirmed Restart with developer chrome', async () => {
    $interfaceMode.set('advanced')
    mount()
    fireEvent.click(statusButton())
    expect(screen.getByRole('button', { name: copy.openSystem })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: en.commandCenter.restartGateway }))
    expect($confirmRequest.get()?.title).toBe(copy.restartTitle)
    expect(mocks.restart).not.toHaveBeenCalled()

    await act(async () => $confirmRequest.get()!.resolve(true))
    expect(mocks.restart).toHaveBeenCalledOnce()
  })

  it('shows Offline: one Reconnect action, no platforms or activity', () => {
    $gatewayState.set('closed')
    mount()
    fireEvent.click(statusButton('Hermes host', copy.offline))
    expect(screen.getAllByText(copy.offline).length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: copy.reconnectGateway })).toBeTruthy()
    expect(screen.queryByText('telegram')).toBeNull()
    expect(screen.queryByRole('button', { name: copy.recentActivity })).toBeNull()
  })

  it('shows another machine only as far as the app knows it, with the actions real for it', () => {
    $connectionsRegistry.set({
      connections: [{ id: 'lab', kind: 'remote', label: 'Lab box', url: 'https://lab.example' }],
      primary: 'local',
      version: 1
    } as never)
    $fleetRoster.set({ agents: [], sources: [{ connectionId: 'lab', kind: 'remote', label: 'Lab box', reachable: false }] })
    mount({ connectionId: 'lab', label: 'Lab box', selected: false })
    fireEvent.click(statusButton('Lab box', copy.offline))
    expect(screen.getByRole('button', { name: copy.checkAgain })).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.switchToMachine })).toBeTruthy()
    expect(screen.queryByText(copy.inferenceLabel)).toBeNull()
  })
})
