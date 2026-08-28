/**
 * #94483: the Scheduled jobs pane read the shared roster with a bare
 * `$lastRoster.get()` while rendering. BotsPane owns the roster fetch, so
 * whenever this pane mounted before the roster hydrated (fresh boot ordering,
 * a renderer reload resetting the atoms) it captured an empty snapshot
 * forever: the pane stayed pinned on its "has to appear in the roster"
 * placeholder even for a focused bot chat whose exact roster row existed, and
 * creating a job silently no-oped.
 *
 * The contract:
 *   1. the pane SUBSCRIBES to the roster, so hydration re-renders it;
 *   2. once the owner resolves it paints real content and a working create
 *      affordance instead of the placeholder;
 *   3. a complete (authoritative) focused owner with no exact roster row STILL
 *      fails closed — the subscription fix must not loosen identity matching.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import type { atom } from 'nanostores'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RoutineJob } from './types'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const { request } = vi.hoisted(() => ({ request: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { atom: nanoAtom } = await import('nanostores')

  return {
    ...sdk,
    host: {
      ...sdk.host,
      request,
      // The pane's owner ladder starts here, so it has to be a real store:
      // `$focusedBotOwner` is resolved once at bot-state module load.
      state: { ...sdk.host.state, focusedSessionOwner: nanoAtom(null) }
    }
  }
})

const { host } = await import('@hermes/plugin-sdk')
const { $lastRoster } = await import('./data')
const { $selectedBot } = await import('./bot-state')
const { RoutinesPane } = await import('./cron')

/** The SDK store the pane's owner ladder reads. */
const $focused = host.state.focusedSessionOwner as unknown as ReturnType<typeof atom>

const job: RoutineJob = { enabled: true, job_id: 'j-1', name: 'Report', schedule: 'every 1h', state: 'scheduled' }

function renderPane() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <RoutinesPane />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  request.mockResolvedValue({ jobs: [job], scoped: 'research' })
  // Mount order under test: the pane paints BEFORE the roster fetch lands.
  $lastRoster.set([])
  $selectedBot.set('default')
  $focused.set({ connectionId: 'local', profile: 'research' })
})

afterEach(() => {
  cleanup()
})

describe('the pane follows the roster hydrating after mount', () => {
  it('starts fail-closed, then paints once the row arrives', async () => {
    renderPane()

    expect(screen.getByText('This agent has to appear in the roster first.')).toBeTruthy()

    // The same focused bot chat now resolves to its exact roster row. Nothing
    // else changes — only a store write the pane must be subscribed to see.
    act(() => $lastRoster.set([{ connectionId: 'local', name: 'research' }]))

    await waitFor(() => expect(screen.queryByText('This agent has to appear in the roster first.')).toBeNull())
    expect(await screen.findByText('Report')).toBeTruthy()
    expect(screen.getByText('Scheduled jobs')).toBeTruthy()
  })

  it('offers a create affordance once the owner resolves', async () => {
    request.mockResolvedValue({ jobs: [], scoped: 'research' })
    renderPane()

    act(() => $lastRoster.set([{ connectionId: 'local', name: 'research' }]))

    // The reported "create silently no-ops" symptom starts here: while the
    // owner is stuck unresolved the pane never offers any create control.
    expect(await screen.findByText('No scheduled jobs yet')).toBeTruthy()

    // Both doors: the header action and the empty state's own call to action.
    const create = screen.getAllByRole('button', { name: 'New cron' })

    expect(create).toHaveLength(2)

    act(() => create[0].click())

    // The dialog opens naming the resolved owner, not a placeholder.
    expect((await screen.findByRole('dialog')).textContent).toMatch(/research/i)
  })
})

describe('identity matching stays exact', () => {
  it('fails closed for a complete focused owner with no roster row', async () => {
    $focused.set({ connectionId: 'local', profile: 'ghost-profile' })
    renderPane()

    act(() => $lastRoster.set([{ connectionId: 'local', name: 'research' }]))

    await waitFor(() => expect(screen.getByText('This agent has to appear in the roster first.')).toBeTruthy())
    expect(request).not.toHaveBeenCalled()
  })
})
