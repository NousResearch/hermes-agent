import { act, cleanup, render } from '@testing-library/react'
import { useRef } from 'react'
import { HashRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { NEW_CHAT_ROUTE, sessionRoute } from '@/app/routes'
import { routeTargetFromToken, sessionContextDrift } from '@/app/session/hooks/session-context-drift'

import { useQuickEntryBridge } from './use-quick-entry-bridge'

const bridge = vi.hoisted(() => ({
  handler: null as null | ((payload: { target: string; text: string }) => void)
}))

vi.mock('@/store/quick-entry', () => ({
  QUICK_TARGET_CURRENT: 'current',
  QUICK_TARGET_NEW: 'new',
  initQuickEntryBridge: () => () => undefined,
  setQuickEntrySubmitHandler: (fn: typeof bridge.handler) => {
    bridge.handler = fn
  }
}))
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return { $gatewayState: atom('open'), $sessions: atom([]) }
})
vi.mock('@/store/session-states', () => ({
  sessionTileDelegate: () => null
}))
vi.mock('@/store/windows', () => ({ isAuxiliaryWindow: () => false }))
vi.mock('@/components/pane-shell/tree/store', () => ({ noteActiveTreeGroup: vi.fn(), revealTreePane: vi.fn() }))
vi.mock('@/contrib/registry', () => ({ registry: { getArea: () => [] } }))

interface Pending {
  startRoute: string
  text: string
  finish: () => void
}

// Real React + the same HashRouter mode as main.tsx; the backend round-trip
// is deferred. The route-ref publication and drift guard match ContribWiring
// and createBackendSessionForSend rather than mocking flushSync or navigation.
function mountHarness(initial = 'old-session') {
  window.history.replaceState(null, '', `#${sessionRoute(initial)}`)
  const pending: Pending[] = []
  const sent: string[] = []
  const aborted: string[] = []

  let navigateTo = (_path: string) => {}

  function Harness() {
    const location = useLocation()
    const navigate = useNavigate()
    navigateTo = navigate
    const routeRef = useRef('')
    routeRef.current = `${location.pathname}:${location.search}:${location.hash}`
    const selectedRef = useRef<string | null>(initial)

    useQuickEntryBridge({
      startFreshSessionDraft: () => {
        navigate(NEW_CHAT_ROUTE)
        selectedRef.current = null
      },
      submitText: async text => {
        const startRoute = routeRef.current
        const startSelected = selectedRef.current
        await new Promise<void>(finish => pending.push({ startRoute, text, finish }))

        const drift = sessionContextDrift({
          startRouteToken: startRoute,
          nowRouteToken: routeRef.current,
          startSelectedStoredId: startSelected,
          nowSelectedStoredId: selectedRef.current
        })

        if (drift) {
          aborted.push(drift)

          return false
        }

        sent.push(text)
        selectedRef.current = `created-${sent.length}`
        navigate(sessionRoute(selectedRef.current))

        return true
      }
    })

    return null
  }

  render(
    <HashRouter useTransitions={false}>
      <Harness />
    </HashRouter>
  )

  return { pending, sent, aborted, navigate: (path: string) => navigateTo(path) }
}

beforeEach(() => {
  vi.clearAllMocks()
})
afterEach(() => {
  cleanup()
  bridge.handler = null
})

describe('Quick Entry new-chat route handoff', () => {
  it('repeated quick entries each submit once after transitioning away from the previous chat', async () => {
    const h = mountHarness()

    for (const text of ['one', 'two', 'three']) {
      await act(async () => {
        bridge.handler!({ target: 'new', text })
      })
      expect(routeTargetFromToken(h.pending.at(-1)!.startRoute)).toBe('__new__')
      await act(async () => {
        h.pending.at(-1)!.finish()
      })
    }

    expect(h.sent).toEqual(['one', 'two', 'three'])
    expect(h.pending).toHaveLength(3)
    expect(h.aborted).toEqual([])
  })

  it('still aborts if the user changes to another chat during the backend round-trip', async () => {
    const h = mountHarness()
    await act(async () => {
      bridge.handler!({ target: 'new', text: 'do not misroute' })
    })
    await act(async () => {
      h.navigate(sessionRoute('unrelated-session'))
    })
    await act(async () => {
      h.pending[0].finish()
    })
    expect(h.sent).toEqual([])
    expect(h.aborted).toEqual(['route:__new__->unrelated-session'])
  })
})
