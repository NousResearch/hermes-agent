import { act, cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'

import type * as RoutesModule from '../routes'

import { ChatRoutesSurface } from './surfaces'

const { modelMenuProps } = vi.hoisted(() => ({
  modelMenuProps: [] as Array<{ gateway?: unknown; profile?: string }>
}))

vi.mock('@/contrib/react/use-contributions', () => ({ useContributions: vi.fn() }))
vi.mock('../chat', () => ({
  ChatView: ({ modelMenuContent }: { modelMenuContent?: ReactNode }) => <>{modelMenuContent}</>
}))
vi.mock('../routes', async importOriginal => {
  const actual = await importOriginal<typeof RoutesModule>()

  return {
    ...actual,
    contributedRoutes: () => [],
    NEW_CHAT_ROUTE: '/',
    ROUTES_AREA: 'routes',
    sessionRoute: (id: string) => `/${id}`
  }
})
vi.mock('../shell/model-menu-panel', () => ({
  ModelMenuPanel: (props: { gateway?: unknown; profile?: string }) => {
    modelMenuProps.push(props)

    return <div data-testid="model-menu" />
  }
}))
vi.mock('./latest-actions', () => ({
  latestChatActions: () => ({}),
  latestSidebarActions: () => ({})
}))

beforeEach(() => {
  modelMenuProps.length = 0
  $activeGatewayProfile.set('alpha')
  $gatewayState.set('open')
})

afterEach(() => {
  cleanup()
  $gateway.set(null)
  $activeGatewayProfile.set('default')
  $gatewayState.set('closed')
})

describe('ChatRoutesSurface active-profile gateway', () => {
  it('replaces the picker transport on an open-to-open profile swap', () => {
    const alphaGateway = { id: 'alpha' } as unknown as HermesGateway
    const betaGateway = { id: 'beta' } as unknown as HermesGateway

    $gateway.set(alphaGateway)

    const actions = {
      getGateway: () => $gateway.get(),
      requestGateway: vi.fn(),
      selectModel: vi.fn()
    }

    render(
      <MemoryRouter initialEntries={['/']}>
        <ChatRoutesSurface actions={actions as never} />
      </MemoryRouter>
    )

    expect(modelMenuProps.at(-1)).toMatchObject({ gateway: alphaGateway, profile: 'alpha' })

    act(() => {
      // A prewarmed profile is already open, so the connection state remains
      // 'open'; only the gateway identity and active profile change.
      $gateway.set(betaGateway)
      $activeGatewayProfile.set('beta')
    })

    expect(modelMenuProps.at(-1)).toMatchObject({ gateway: betaGateway, profile: 'beta' })
  })
})