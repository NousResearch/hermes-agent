import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import { setActiveSessionId, setBusy, setCurrentUsage, setSelectedStoredSessionId } from '@/store/session'
import { $sessionTiles, clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { useStatusbarItems } from './use-statusbar-items'

const SESSION_ID = 'session-1'
const TILE_RUNTIME_ID = 'tile-runtime-1'
const TILE_STORED_ID = 'tile-stored-1'

let statusbarItems: readonly StatusbarItem[] = []

const noop = () => {}

const requestGateway = <T = unknown,>() => new Promise<T>(() => {})

function Harness() {
  const items = useStatusbarItems({
    agentsOpen: false,
    chatOpen: true,
    commandCenterOpen: false,
    extraLeftItems: [],
    extraRightItems: [],
    freshDraftReady: true,
    gatewayState: 'open',
    inferenceStatus: null,
    openAgents: noop,
    openCommandCenterSection: noop,
    requestGateway,
    statusSnapshot: null,
    toggleCommandCenter: noop
  })

  statusbarItems = items.statusbarItems

  return null
}

function contextUsageItem(): StatusbarItem {
  const item = statusbarItems.find(candidate => candidate.id === 'context-usage')

  if (!item) {
    throw new Error('context usage statusbar item was not registered')
  }

  return item
}

afterEach(() => {
  cleanup()
  statusbarItems = []
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 })
  clearAllSessionStates()
  $sessionTiles.set([])
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  vi.restoreAllMocks()
})

describe('context usage statusbar freshness', () => {
  it('replaces the last completed occupancy with unknown while the focused turn is running', () => {
    setActiveSessionId(SESSION_ID)
    setBusy(false)
    setCurrentUsage({
      calls: 1,
      context_max: 272_000,
      context_percent: 100,
      context_used: 371_176,
      input: 371_176,
      output: 0,
      total: 371_176
    })

    render(<Harness />)

    expect(contextUsageItem()).toMatchObject({
      detail: '[██████████] 100%',
      label: '371.2k/272k'
    })

    act(() => setBusy(true))

    expect(contextUsageItem()).toMatchObject({
      detail: 'unknown',
      label: 'Context usage'
    })

    act(() => {
      setCurrentUsage({
        calls: 2,
        context_max: 272_000,
        context_percent: 59,
        context_used: 159_956,
        input: 159_956,
        output: 0,
        total: 159_956
      })
      setBusy(false)
    })

    expect(contextUsageItem()).toMatchObject({
      detail: '[██████░░░░] 59%',
      label: '160k/272k'
    })
  })

  it('uses the focused tile busy state instead of leaking the idle primary usage', () => {
    setActiveSessionId(SESSION_ID)
    setSelectedStoredSessionId(SESSION_ID)
    setBusy(false)
    setCurrentUsage({
      calls: 1,
      context_max: 272_000,
      context_percent: 25,
      context_used: 68_000,
      input: 68_000,
      output: 0,
      total: 68_000
    })

    $sessionTiles.set([{ runtimeId: TILE_RUNTIME_ID, storedSessionId: TILE_STORED_ID }])
    $layoutTree.set(group([`session-tile:${TILE_STORED_ID}`], { id: 'tile-group' }))
    noteActiveTreeGroup('tile-group')
    publishSessionState(TILE_RUNTIME_ID, {
      ...createClientSessionState(TILE_STORED_ID),
      busy: true,
      usage: {
        calls: 1,
        context_max: 272_000,
        context_percent: 100,
        context_used: 371_176,
        input: 371_176,
        output: 0,
        total: 371_176
      }
    })

    render(<Harness />)

    expect(contextUsageItem()).toMatchObject({
      detail: 'unknown',
      label: 'Context usage'
    })

    act(() =>
      publishSessionState(TILE_RUNTIME_ID, {
        ...createClientSessionState(TILE_STORED_ID),
        busy: false,
        usage: {
          calls: 2,
          context_max: 272_000,
          context_percent: 59,
          context_used: 159_956,
          input: 159_956,
          output: 0,
          total: 159_956
        }
      })
    )

    expect(contextUsageItem()).toMatchObject({
      detail: '[██████░░░░] 59%',
      label: '160k/272k'
    })
  })
})
