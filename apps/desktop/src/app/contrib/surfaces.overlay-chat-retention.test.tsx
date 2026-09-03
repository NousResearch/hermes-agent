import { act, cleanup, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter, type NavigateFunction, useNavigate } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { OVERLAY_ROUTE_PATHS } from '@/app/routes'

import { ChatRoutesSurface } from './surfaces'
import type { WiringActions } from './types'

const counters = vi.hoisted(() => ({ mounts: 0 }))

// A sentinel ChatView that reports its own MOUNT count: "the chat stays behind
// the overlay" means the very same instance survives, not that a fresh one is
// rendered — a remount is what tears down the transcript and composer state.
vi.mock('../chat', () => ({
  ChatView: function MockChatView() {
    useEffect(() => {
      counters.mounts += 1
    }, [])

    return <div data-testid="chat-view" />
  }
}))

vi.mock('../skills', () => ({ SkillsView: () => <div data-testid="skills-view" /> }))
vi.mock('../messaging', () => ({ MessagingView: () => <div data-testid="messaging-view" /> }))
vi.mock('../artifacts', () => ({ ArtifactsView: () => <div data-testid="artifacts-view" /> }))

// ChatRoutesSurface only forwards these through `latestChatActions`; the mocked
// ChatView never invokes them, so a bare stand-in keeps the test to the routing
// behaviour under test.
const actions = {
  getGateway: () => null,
  openAgents: () => undefined,
  openCommandCenterSection: () => undefined,
  requestGateway: () => undefined,
  selectModel: () => undefined,
  toggleCommandCenter: () => undefined
} as unknown as WiringActions

function mountAt(pathname: string) {
  return render(
    <MemoryRouter initialEntries={[pathname]}>
      <ChatRoutesSurface actions={actions} />
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
  counters.mounts = 0
})

describe('workspace route table — chat retention behind overlays', () => {
  it.each([...OVERLAY_ROUTE_PATHS])('keeps the workspace chat mounted at /%s', path => {
    mountAt(`/${path}`)

    expect(screen.queryByTestId('chat-view')).not.toBeNull()
  })

  it('still replaces the chat with a full-page workspace route', () => {
    mountAt('/skills')

    expect(screen.queryByTestId('chat-view')).toBeNull()
  })

  it('reconciles rather than remounts the chat when an overlay opens and closes', () => {
    let navigate: NavigateFunction | undefined

    function CaptureNavigate() {
      const to = useNavigate()

      useEffect(() => {
        navigate = to
      }, [to])

      return null
    }

    render(
      <MemoryRouter initialEntries={['/session-a']}>
        <CaptureNavigate />
        <ChatRoutesSurface actions={actions} />
      </MemoryRouter>
    )

    expect(counters.mounts).toBe(1)

    for (const path of OVERLAY_ROUTE_PATHS) {
      act(() => navigate!(`/${path}`))
      expect(screen.queryByTestId('chat-view')).not.toBeNull()
    }

    act(() => navigate!('/session-a'))

    // One mount for the whole tour: opening every overlay in turn never tore
    // the live chat down.
    expect(counters.mounts).toBe(1)
  })
})
