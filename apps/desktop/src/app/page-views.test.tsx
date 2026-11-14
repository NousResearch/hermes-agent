import { act, cleanup, render, screen } from '@testing-library/react'
import { Suspense } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import {
  ArtifactsView,
  CapabilitiesView,
  CronView,
  preloadRailPagesInRedesign,
  preloadRailPagesWhenIdle,
  ProfilesView,
  SettingsView
} from './page-views'

const nested = vi.hoisted(() => ({
  directory: vi.fn(async () => undefined),
  agentSettings: vi.fn(async () => undefined),
  agentOverview: vi.fn(async () => undefined),
  capabilities: vi.fn(async () => undefined),
  artifacts: vi.fn(async () => undefined)
}))

const page = (name: string) => () => <h1>{name}</h1>

vi.mock('./capabilities', () => ({ CapabilitiesView: page('Capabilities') }))
vi.mock('./artifacts', () => ({ ArtifactsView: page('Library') }))
vi.mock('./cron', () => ({ CronView: page('Schedules') }))
vi.mock('./profiles', () => ({ ProfilesView: page('Agents'), preloadAgentDirectory: nested.directory }))
vi.mock('./agent-settings/entry', () => ({
  SettingsView: page('Settings'),
  preloadAgentSettingsView: nested.agentSettings
}))

vi.mock('./agent-settings/overview-queries', () => ({ prefetchLandingAgentOverview: nested.agentOverview }))
vi.mock('./capabilities/prefetch', () => ({ prefetchCapabilitiesInventory: nested.capabilities }))
vi.mock('@/store/item-index', () => ({ refreshItemIndex: nested.artifacts }))

const idle = new Map<number, () => void>()
let nextId = 0

beforeEach(() => {
  vi.clearAllMocks()
  idle.clear()
  vi.stubGlobal('requestIdleCallback', (task: () => void) => (idle.set(++nextId, task), nextId))
  vi.stubGlobal('cancelIdleCallback', (id: number) => idle.delete(id))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  $shellChoice.set(null)
})

async function drainIdle() {
  for (let entry = idle.entries().next(); !entry.done; entry = idle.entries().next()) {
    const [id, task] = entry.value
    idle.delete(id)
    await act(async () => task())
  }
}

describe('rail page preloading', () => {
  it('warms every rail destination while idle, so a click paints the page without suspending', async () => {
    preloadRailPagesWhenIdle()
    await drainIdle()

    for (const [View, title] of [
      [ProfilesView, 'Agents'],
      [CapabilitiesView, 'Capabilities'],
      [ArtifactsView, 'Library'],
      [CronView, 'Schedules'],
      [SettingsView, 'Settings']
    ] as const) {
      // Synchronous render, no act: what a rail click commits in its own frame.
      render(
        <Suspense fallback={<p>fallback</p>}>
          <View onClose={() => undefined} />
        </Suspense>
      )

      expect(screen.getByRole('heading', { name: title })).toBeTruthy()
      expect(screen.queryByText('fallback')).toBeNull()
      cleanup()
    }

    // The Agents page's nested chunks are warmed too, not just its entry.
    expect(nested.directory).toHaveBeenCalled()
    expect(nested.agentSettings).toHaveBeenCalled()
    // And the data those pages lay out from, so they paint final rows instead of placeholders.
    expect(nested.agentOverview).toHaveBeenCalled()
    expect(nested.capabilities).toHaveBeenCalled()
    expect(nested.artifacts).toHaveBeenCalled()
  })

  // Classic has no rail: like main, it loads a page and its reads (the artifact index fetches every
  // recent transcript) only when the page opens.
  it('warms nothing at launch in classic, and starts once the window flips to the redesign', async () => {
    $shellChoice.set(false)
    const stop = preloadRailPagesInRedesign()
    await drainIdle()

    expect(nested.artifacts).not.toHaveBeenCalled()
    expect(nested.directory).not.toHaveBeenCalled()

    $shellChoice.set(true)
    await drainIdle()
    stop()

    expect(nested.artifacts).toHaveBeenCalled()
  })

  it('stops warming when the shell unmounts', async () => {
    preloadRailPagesWhenIdle()()
    await drainIdle()

    expect(nested.directory).not.toHaveBeenCalled()
  })
})
