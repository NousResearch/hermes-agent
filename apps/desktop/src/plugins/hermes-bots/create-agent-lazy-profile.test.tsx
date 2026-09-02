/**
 * New Bot creates its profile LAZILY. Two contracts hang off that:
 *
 *  - MCP OAuth / API-key credentials must be configurable DURING creation
 *    rather than forcing create-then-edit, so the setup button materializes
 *    the profile through `ensureAgentCreated()` on its first action. The
 *    creation is single-flighted and idempotent: whichever door fires first
 *    (a setup click, the Capabilities tab, or Create Bot itself) is the only
 *    `profiles.create` the dialog ever sends.
 *  - `reset()` has to restore the SAME clone source the dialog mounted with.
 *    It restored '__none__' instead: the first open cloned the main profile,
 *    every later open after a create or cancel silently started fresh.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DataModule from './data'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

interface SkillsViewProps {
  fixedConnection?: string
  fixedProfile?: string
}

const mocks = vi.hoisted(() => ({
  connections: vi.fn(async () => [] as { id: string; label: string }[]),
  createCanonicalChat: vi.fn(async () => 'session-1'),
  deleteBot: vi.fn(async () => undefined),
  /** Flipped off to model a desktop build that predates the live surface. */
  hasSkillsView: { value: true },
  notify: vi.fn(),
  notifyError: vi.fn(),
  request: vi.fn(),
  requestProfile: vi.fn(async () => ({})),
  saveBotMeta: vi.fn(),
  skillsView: [] as SkillsViewProps[]
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  const SkillsViewStub = (props: SkillsViewProps) => {
    mocks.skillsView.push(props)

    return null
  }

  // Builds that route `fixedConnection` get the live Capabilities tab for
  // remote targets too, pinned to the target machine's backend.
  SkillsViewStub.supportsFixedConnection = true

  const mocked: Record<string, unknown> = {
    ...original,
    host: {
      ...original.host,
      connections: mocks.connections,
      notify: mocks.notify,
      notifyError: mocks.notifyError,
      request: mocks.request,
      requestProfile: mocks.requestProfile
    },
    // The plugin bundle normally lands via `ctx.i18n.register` at load.
    usePluginI18n: () => translateBots
  }

  Object.defineProperty(mocked, 'SkillsView', {
    configurable: true,
    enumerable: true,
    get: () => (mocks.hasSkillsView.value ? SkillsViewStub : undefined)
  })

  return mocked
})

vi.mock('./canonical-chat', () => ({ createCanonicalChat: mocks.createCanonicalChat }))
vi.mock('./profile-ops', () => ({ deleteBot: mocks.deleteBot }))
vi.mock('./data', async importOriginal => {
  const original = await importOriginal<typeof DataModule>()

  return { ...original, saveBotMeta: mocks.saveBotMeta }
})

// Each build re-imports the dialog's whole module graph.
vi.setConfig({ testTimeout: 30_000 })

const roster: RosterRow[] = [{ connectionId: 'local', name: 'default' }]

/** A catalog entry that needs an API key — the only row that renders setup. */
const catalog = { auth: null, installed: false, name: 'linear', requires: ['LINEAR_API_KEY'] }

function withQueryClient(children: ReactNode) {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {children}
    </QueryClientProvider>
  )
}

async function renderDialog(hasSkillsView: boolean) {
  mocks.hasSkillsView.value = hasSkillsView
  vi.resetModules()

  const { CreateAgentDialog } = await import('./create-dialog')

  const view = render(withQueryClient(<CreateAgentDialog onClose={() => undefined} open roster={roster} />))

  fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
  fireEvent.click(screen.getByRole('button', { name: /Advanced/ }))

  return view
}

function createCalls() {
  return mocks.request.mock.calls.filter(([method]) => method === 'profiles.create')
}

/** The control under a `labeled(...)` caption — the label is presentational,
 *  so it carries no `for`/`id` pair to query by. */
function controlUnder(caption: string) {
  return screen.getByText(caption).parentElement!.querySelector('[role="combobox"]') as HTMLElement
}

beforeAll(async () => {
  // Radix Select/Dialog reach for APIs jsdom does not implement.
  Element.prototype.scrollIntoView = () => undefined
  Element.prototype.hasPointerCapture = () => false
  Element.prototype.releasePointerCapture = () => undefined
  Element.prototype.setPointerCapture = () => undefined
  // Warm the transform cache so the first test isn't charged for it.
  await import('./create-dialog')
}, 120_000)

beforeEach(() => {
  vi.clearAllMocks()
  mocks.skillsView.length = 0
  mocks.connections.mockResolvedValue([])
  mocks.requestProfile.mockResolvedValue({})
  mocks.request.mockImplementation(async (method: string) => {
    switch (method) {
      case 'mcp.catalog':
        return { servers: [catalog] }

      case 'mcp.servers.test':
        return { ok: true }

      case 'profiles.describe':
        return { mcp_servers: [], skills: [], toolsets: [] }

      default:
        return {}
    }
  })
})

afterEach(() => {
  cleanup()
})

describe('materializing the draft profile', () => {
  it('creates it once when the Capabilities tab opens, pinned to the new slug', async () => {
    await renderDialog(true)

    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))

    await waitFor(() => expect(createCalls()).toHaveLength(1))
    expect(createCalls()[0][1]).toMatchObject({ clone_from: 'default', name: 'inbox-triage' })
    // The live surface needs a REAL backend row to write to.
    await waitFor(() => expect(mocks.skillsView.at(-1)).toMatchObject({ fixedProfile: 'inbox-triage' }))

    // Create Bot goes through the same helper — no duplicate profiles.create.
    fireEvent.click(screen.getByRole('button', { name: 'Create Bot' }))

    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(createCalls()).toHaveLength(1)
  })

  it('pins a remote-target draft to the TARGET machine, not the active gateway', async () => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])

    await renderDialog(true)

    // "Create on" only exists on a multi-connection desktop.
    await screen.findByText('Create on')
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))

    // The create lands on THAT machine's backend — no gateway switch, and the
    // clone source is the remote's own default (a local profile name the
    // remote box does not have would fail there).
    await waitFor(() =>
      expect(mocks.requestProfile).toHaveBeenCalledWith(
        expect.objectContaining({ connectionId: 'studio' }),
        'profiles.create',
        expect.objectContaining({ clone_from: 'default', name: 'inbox-triage' })
      )
    )
    expect(createCalls()).toHaveLength(0)
    await waitFor(() =>
      expect(mocks.skillsView.at(-1)).toMatchObject({ fixedConnection: 'studio', fixedProfile: 'inbox-triage' })
    )
  })

  it('creates it on the first MCP setup click, then adds the server to it', async () => {
    // Older builds keep the staged MCP tab, which is where the setup button
    // lives. There is no "save the agent first" gate: the button is live.
    await renderDialog(false)

    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))

    await waitFor(() => expect(createCalls()).toHaveLength(1))
    expect(createCalls()[0][1]).toMatchObject({ name: 'inbox-triage' })
    // The add lands on the profile the click just minted, not on `null`.
    await waitFor(() =>
      expect(mocks.request).toHaveBeenCalledWith('mcp.servers.add', {
        name: 'linear',
        preset: 'linear',
        profile: 'inbox-triage'
      })
    )
  })
})

describe('the clone-from default', () => {
  it('regression: reset restores the mounted default, not "fresh profile"', async () => {
    await renderDialog(true)

    // `labeled()` renders a bare <label>, so the trigger is found by position.
    const cloneFrom = () =>
      screen.getByText('Clone from profile').parentElement!.querySelector('[role="combobox"]') as HTMLElement

    expect(cloneFrom().textContent).toBe('default')

    fireEvent.click(cloneFrom())
    fireEvent.click(await screen.findByRole('option', { name: /Fresh profile/ }))
    expect(cloneFrom().textContent).toMatch(/Fresh profile/)

    fireEvent.click(screen.getByRole('button', { name: 'Create Bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())

    // Second open: the picker must read `default` again, or every agent after
    // the first silently starts from a bare profile.
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }))
    expect(cloneFrom().textContent).toBe('default')
  })
})
