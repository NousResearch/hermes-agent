import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { Suspense } from 'react'
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import type * as SessionStore from '@/store/session'

// @vitest-environment jsdom

const mocks = vi.hoisted(() => ({
  createWorld: vi.fn(),
  destroy: vi.fn(),
  loadManifest: vi.fn(),
  openSession: vi.fn(),
  setSessionOwnerHint: vi.fn()
}))

vi.mock('./manifest', () => ({ loadWorldManifest: mocks.loadManifest }))
vi.mock('./world/create-world', () => ({ createLunarCityWorld: mocks.createWorld }))
vi.mock('@/app/open-session', () => ({ openSession: mocks.openSession }))
vi.mock('@/store/session', async importOriginal => ({
  ...(await importOriginal<typeof SessionStore>()),
  setSessionOwnerHint: mocks.setSessionOwnerHint
}))

import {
  LUNAR_CITY_CONTRIBUTIONS,
  LUNAR_CITY_NAV_ORDER,
  LUNAR_CITY_ROUTE,
  LunarCityRoute,
  openEntitySession,
  openLeaderFullChat
} from './contribution'

function KanbanRoute() {
  const navigate = useNavigate()

  return (
    <div>
      <h1>Kanban</h1>
      <button onClick={() => navigate(LUNAR_CITY_ROUTE)} type="button">
        Open Lunar City
      </button>
    </div>
  )
}

function RouteHarness({ initialEntry = '/kanban' }: { initialEntry?: string }) {
  return (
    <MemoryRouter initialEntries={[initialEntry]}>
      <Suspense fallback={<p>Loading route</p>}>
        <Routes>
          <Route element={<KanbanRoute />} path="kanban" />
          <Route element={<LunarCityRoute />} path="lunar-city" />
          <Route element={<h1>Memory graph</h1>} path="starmap" />
        </Routes>
      </Suspense>
    </MemoryRouter>
  )
}

beforeEach(() => {
  mocks.createWorld.mockReset().mockResolvedValue({
    applySnapshot: vi.fn(),
    destroy: mocks.destroy,
    leaderStateClips: {},
    qualityMode: 'low',
    sendIntent: vi.fn(),
    setCameraMode: vi.fn(),
    setQualityMode: vi.fn()
  })
  mocks.destroy.mockReset()
  mocks.loadManifest.mockReset().mockResolvedValue({ characterAssets: { leaders: [] } })
})

describe('Lunar City route contribution', () => {
  it('hands a leader conversation to the ordinary full chat with its exact owner hint', () => {
    const navigate = vi.fn()
    const owner = { connectionId: 'source-a', profile: 'owl' }

    openLeaderFullChat('stored-owl', owner, navigate)

    expect(mocks.setSessionOwnerHint).toHaveBeenCalledWith('stored-owl', owner)
    expect(mocks.openSession).toHaveBeenCalledWith('stored-owl', navigate, 'main')
  })

  it('opens an inspected worker session on its complete exact owner route', () => {
    const navigate = vi.fn()

    const target = {
      connectionId: 'source-b',
      profile: 'builder',
      sessionId: 'runtime-9',
      storedSessionId: 'stored-9'
    }

    openEntitySession(target, navigate)

    expect(mocks.setSessionOwnerHint).toHaveBeenCalledWith('stored-9', {
      connectionId: 'source-b',
      profile: 'builder'
    })
    expect(mocks.openSession).toHaveBeenCalledWith('stored-9', navigate, 'main')
  })

  it('registers its dedicated destination directly after the order-50 Kanban entry', () => {
    const page = LUNAR_CITY_CONTRIBUTIONS.find(contribution => contribution.area === 'routes')
    const nav = LUNAR_CITY_CONTRIBUTIONS.find(contribution => contribution.area === 'sidebar.nav')

    expect(page?.data).toEqual({ path: '/lunar-city' })
    expect(nav).toMatchObject({
      order: 60,
      data: { codicon: 'globe', label: 'Lunar City', path: '/lunar-city' }
    })
    expect(LUNAR_CITY_NAV_ORDER).toBeGreaterThan(50)

    const testArea = 'test.lunar-city.sidebar-order'

    const dispose = registry.registerMany([
      { id: 'kanban', area: testArea, order: 50, data: { label: 'Kanban' } },
      { ...nav!, area: testArea }
    ])

    expect(registry.getArea(testArea).map(item => (item.data as { label: string }).label)).toEqual([
      'Kanban',
      'Lunar City'
    ])
    dispose()
  })

  it('does not initialize the world during normal Kanban operation, then destroys it when navigation leaves', async () => {
    render(<RouteHarness />)

    expect(screen.getByRole('heading', { name: 'Kanban' })).toBeTruthy()
    expect(mocks.loadManifest).not.toHaveBeenCalled()
    expect(mocks.createWorld).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Open Lunar City' }))
    await waitFor(() => expect(mocks.createWorld).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByRole('button', { name: 'Open memory graph' }))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Memory graph' })).toBeTruthy())
    expect(mocks.destroy).toHaveBeenCalledOnce()
  })

  it('initializes only when its dedicated route is opened directly', async () => {
    render(<RouteHarness initialEntry={LUNAR_CITY_ROUTE} />)

    await waitFor(() => expect(mocks.loadManifest).toHaveBeenCalledOnce())
    await waitFor(() => expect(mocks.createWorld).toHaveBeenCalledOnce())
  })
})
