import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { resetAllBindings, setBinding } from '@/store/keybinds'
import { WORKSTREAM_PANE_ID } from '@/store/layout'
import { $paneStates } from '@/store/panes'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import {
  $workstreamFilter,
  $workstreamVisibleSessionIds,
  setWorkstreamVisibleSessionIds
} from '@/store/workstream-filter'
import type { SessionInfo } from '@/types/hermes'

import { useKeybinds } from './use-keybinds'

const session = (id: string): SessionInfo =>
  ({
    archived: false,
    cwd: null,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: id,
    tool_call_count: 0
  }) as SessionInfo

function LocationProbe() {
  const location = useLocation()

  return <output aria-label="location">{location.pathname}</output>
}

interface HarnessProps {
  startFreshSession?: () => void
}

function Harness({ startFreshSession = vi.fn() }: HarnessProps) {
  useKeybinds({
    startFreshSession,
    toggleCommandCenter: vi.fn(),
    toggleSelectedPin: vi.fn()
  })

  return (
    <>
      <input aria-label="composer" />
      <aside data-workstream-progress-rail tabIndex={-1} />
      <LocationProbe />
    </>
  )
}

const renderHarness = (props: HarnessProps = {}) =>
  render(
    <MemoryRouter initialEntries={['/b']}>
      <Harness {...props} />
    </MemoryRouter>
  )

describe('useKeybinds workstream actions', () => {
  beforeEach(() => {
    resetAllBindings()
    $sessions.set([session('a'), session('b'), session('c')])
    $selectedStoredSessionId.set('b')
    $workstreamFilter.set('all')
    $workstreamVisibleSessionIds.set([])
    setWorkstreamVisibleSessionIds(['a', 'b', 'c'])
    $paneStates.set({ [WORKSTREAM_PANE_ID]: { open: false } })
  })

  afterEach(() => {
    cleanup()
    resetAllBindings()
    $sessions.set([])
    $selectedStoredSessionId.set(null)
    $workstreamVisibleSessionIds.set([])
  })

  it('navigates to the next visible workstream from an editable target', async () => {
    renderHarness()

    fireEvent.keyDown(screen.getByLabelText('composer'), { altKey: true, code: 'KeyJ', key: 'j' })

    await waitFor(() => expect(screen.getByLabelText('location').textContent).toBe('/c'))
  })

  it('navigates to the previous visible workstream from an editable target', async () => {
    renderHarness()

    fireEvent.keyDown(screen.getByLabelText('composer'), { altKey: true, code: 'KeyK', key: 'k' })

    await waitFor(() => expect(screen.getByLabelText('location').textContent).toBe('/a'))
  })

  it('cycles the workstream filter from an editable target', () => {
    renderHarness()

    fireEvent.keyDown(screen.getByLabelText('composer'), { altKey: true, code: 'KeyM', key: 'm' })

    expect($workstreamFilter.get()).toBe('active')
  })

  it('opens and focuses the workstream pane from an editable target', async () => {
    renderHarness()

    fireEvent.keyDown(screen.getByLabelText('composer'), { altKey: true, code: 'Comma', key: ',' })

    expect($paneStates.get()[WORKSTREAM_PANE_ID]?.open).toBe(true)
    await waitFor(() => expect(document.activeElement).toBe(document.querySelector('[data-workstream-progress-rail]')))
  })

  it('does not allow arbitrary alt-bound actions from an editable target', () => {
    const startFreshSession = vi.fn()
    setBinding('session.new', ['alt+x'])
    renderHarness({ startFreshSession })

    fireEvent.keyDown(screen.getByLabelText('composer'), { altKey: true, code: 'KeyX', key: 'x' })

    expect(startFreshSession).not.toHaveBeenCalled()
  })
})
