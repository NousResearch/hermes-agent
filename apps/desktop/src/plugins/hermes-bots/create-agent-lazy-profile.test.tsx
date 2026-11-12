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
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DataModule from './data'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

interface CapabilitiesViewProps {
  fixedConnection?: string
  fixedProfile?: string
}

const mocks = vi.hoisted(() => ({
  completeMcpOAuth: vi.fn(),
  connections: vi.fn(async () => [] as { id: string; label: string }[]),
  createCanonicalChat: vi.fn(async () => 'session-1'),
  deleteBot: vi.fn(async () => undefined),
  dropLeftoverProfilePrefs: vi.fn(),
  /** Flipped off to model a desktop build that predates the live surface. */
  hasCapabilitiesView: { value: true },
  notify: vi.fn(),
  notifyError: vi.fn(),
  request: vi.fn(),
  requestProfile: vi.fn(async (_route: unknown, _method: string, _params?: unknown) => ({})),
  saveBotMeta: vi.fn(),
  skillsView: [] as CapabilitiesViewProps[]
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  const CapabilitiesViewStub = (props: CapabilitiesViewProps) => {
    mocks.skillsView.push(props)

    return null
  }

  // Builds that route `fixedConnection` get the live Capabilities tab for
  // remote targets too, pinned to the target machine's backend.
  CapabilitiesViewStub.supportsFixedConnection = true

  const mocked: Record<string, unknown> = {
    ...original,
    host: {
      ...original.host,
      completeMcpOAuth: mocks.completeMcpOAuth,
      connections: mocks.connections,
      dropLeftoverProfilePrefs: mocks.dropLeftoverProfilePrefs,
      notify: mocks.notify,
      notifyError: mocks.notifyError,
      request: mocks.request,
      requestProfile: mocks.requestProfile
    },
    // The plugin bundle normally lands via `ctx.i18n.register` at load.
    usePluginI18n: () => translateBots
  }

  Object.defineProperty(mocked, 'CapabilitiesView', {
    configurable: true,
    enumerable: true,
    get: () => (mocks.hasCapabilitiesView.value ? CapabilitiesViewStub : undefined)
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

async function renderDialog(hasCapabilitiesView: boolean, onConfigureModel?: (bot: RosterRow) => void) {
  mocks.hasCapabilitiesView.value = hasCapabilitiesView
  vi.resetModules()

  const { CreateAgentDialog } = await import('./create-dialog')

  const view = render(
    withQueryClient(
      <CreateAgentDialog onClose={() => undefined} onConfigureModel={onConfigureModel} open roster={roster} />
    )
  )

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
  mocks.notify.mockReset()
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

afterEach(async () => {
  cleanup()
  // Radix's focus scope dispatches its unmount event from a 0 ms timer; let it fire
  // here, or it lands in the next test file's jsdom window and throws there.
  await new Promise(resolve => setTimeout(resolve, 0))
})

describe('materializing the draft profile', () => {
  it.each([
    ['小助手', 'u5c0f-u52a9-u624b'],
    ['test机器人', 'test-u673a-u5668-u4eba']
  ])('creates %s with an ASCII profile id and retains the display name', async (enteredName, profileName) => {
    await renderDialog(true)

    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), {
      target: { value: enteredName }
    })

    const create = screen.getByRole('button', { name: 'Create bot' })

    expect((create as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(create)

    await waitFor(() => expect(createCalls()).toHaveLength(1))
    expect(createCalls()[0][1]).toMatchObject({
      description: enteredName,
      name: profileName
    })
    expect(mocks.saveBotMeta).toHaveBeenCalledWith(profileName, expect.objectContaining({ title: enteredName }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith(profileName, { kickoff: true }))
  })

  it('creates it once when the Capabilities tab opens, pinned to the new slug', async () => {
    await renderDialog(true)

    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))

    await waitFor(() => expect(createCalls()).toHaveLength(1))
    expect(createCalls()[0][1]).toMatchObject({ clone_from: 'default', name: 'inbox-triage' })
    // The live surface needs a REAL backend row to write to.
    await waitFor(() => expect(mocks.skillsView.at(-1)).toMatchObject({ fixedProfile: 'inbox-triage' }))

    // Create Bot goes through the same helper — no duplicate profiles.create.
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(createCalls()).toHaveLength(1)
  })

  it('creates it once a name is typed AFTER the Capabilities tab was opened nameless', async () => {
    await renderDialog(true)
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await screen.findByText(/Name the bot first/)

    // Mid-word keystrokes must not each mint (and orphan) a draft.
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'late' } })
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'late-name' } })
    // Edited during the debounce: the draft must carry it, not the value from the last name change.
    fireEvent.change(screen.getByPlaceholderText('Inbox Triage'), { target: { value: 'Late Title' } })

    await waitFor(() => expect(mocks.skillsView.at(-1)).toMatchObject({ fixedProfile: 'late-name' }), { timeout: 3000 })
    expect(createCalls().map(([, params]) => params.name)).toEqual(['late-name'])
    expect(createCalls()[0][1].description).toBe('Late Title')
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

  it('lets a remote-target Bot start fresh instead of forcing a clone of the target default', async () => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])

    await renderDialog(true)

    await screen.findByText('Create on')
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))

    // The picker stays live for a remote target and offers only what exists
    // on THAT machine: fresh, or its own default — never a local roster name.
    const cloneFrom = controlUnder('Clone from profile (on Studio)')

    expect(cloneFrom.getAttribute('data-disabled')).toBeNull()
    fireEvent.click(cloneFrom)
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual([
      'Fresh profile (bundled skills)',
      'default'
    ])
    fireEvent.click(screen.getByRole('option', { name: /Fresh profile/ }))
    // Shared keys live on the TARGET machine, and the label says which.
    screen.getByText('Share keys & accounts with the default profile on Studio')

    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() =>
      expect(mocks.requestProfile).toHaveBeenCalledWith(
        expect.objectContaining({ connectionId: 'studio' }),
        'profiles.create',
        expect.objectContaining({ clone_from: null, name: 'inbox-triage' })
      )
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

// A create never lands on a live name, so a folder or startup profile stored
// under the new bot's name was left by a deleted profile. Bot Mode creates
// through its own RPC, so it has to drop them the way the profile dialog does.
describe('leftover prefs under the new bot name', () => {
  it('drops them once the create succeeded', async () => {
    let created = false
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) => {
      if (method === 'profiles.create') {
        created = true
      }

      return base(method, params)
    })
    mocks.dropLeftoverProfilePrefs.mockImplementationOnce(() => expect(created).toBe(true))

    await renderDialog(true)
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(mocks.dropLeftoverProfilePrefs.mock.calls).toEqual([['inbox-triage', null]])
  })

  it('names the machine a remote bot was created on', async () => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])

    await renderDialog(true)
    await screen.findByText('Create on')
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() => expect(mocks.dropLeftoverProfilePrefs).toHaveBeenCalled())
    expect(mocks.dropLeftoverProfilePrefs.mock.calls).toEqual([['inbox-triage', 'studio']])
  })

  it('keeps them when the create fails', async () => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) => {
      if (method === 'profiles.create') {
        throw new Error('disk full')
      }

      return base(method, params)
    })

    await renderDialog(true)
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() => expect(createCalls()).toHaveLength(1))
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(mocks.dropLeftoverProfilePrefs).not.toHaveBeenCalled()
  })
})

// #141623: the draft profile is minted before the dialog finishes. Cancelling
// must delete the draft THIS dialog created, on the machine it created it on,
// including one whose create or configure is still in flight when the user
// backs out — it used to survive as a bot.
describe('cancelling New Bot discards its draft profile', () => {
  it('deletes the draft whose profiles.create was still in flight at cancel', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishCreate!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) =>
      method === 'profiles.create'
        ? new Promise(resolve => {
            finishCreate = () => resolve({})
          })
        : base(method, params)
    )

    await renderDialog(true)
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(createCalls()).toHaveLength(1))

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    finishCreate()

    await waitFor(() => expect(mocks.deleteBot).toHaveBeenCalledWith({ name: 'inbox-triage' }))
    // The abandoned draft is not configured or adopted as a finished bot.
    expect(mocks.saveBotMeta).not.toHaveBeenCalled()
    expect(mocks.createCanonicalChat).not.toHaveBeenCalled()
  })

  // The draft lives where it was CREATED. Moving "Create on" afterwards must not
  // send the delete to the newly picked machine, where a real bot of the same
  // name may live, and must not leave the draft behind on its own machine.
  it('discards a local draft locally when "Create on" moves to another machine', async () => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])

    await renderDialog(true)
    await screen.findByText('Create on')
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(mocks.skillsView.at(-1)).toMatchObject({ fixedProfile: 'inbox-triage' }))

    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect(mocks.deleteBot).toHaveBeenCalledWith({ name: 'inbox-triage' }))
    expect(mocks.requestProfile).not.toHaveBeenCalledWith(expect.anything(), 'cli.exec', expect.anything())
  })

  it('discards a remote draft on its own machine when "Create on" moves back', async () => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])

    await renderDialog(true)
    await screen.findByText('Create on')
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() =>
      expect(mocks.skillsView.at(-1)).toMatchObject({ fixedConnection: 'studio', fixedProfile: 'inbox-triage' })
    )

    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: /This Mac/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() =>
      expect(mocks.requestProfile).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'studio' }), 'cli.exec', {
        argv: ['profile', 'delete', 'inbox-triage', '--yes']
      })
    )
    expect(mocks.deleteBot).not.toHaveBeenCalled()
  })

  // The draft exists once profiles.create returns, but the flight still awaits
  // the capability configure. A Cancel (or the dialog going away) in that window
  // deletes the draft; the flight must not then record bot meta for it.
  it.each([
    ['Cancel', () => fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))],
    ['unmount', () => cleanup()]
  ])('writes no bot meta for a draft abandoned (%s) while its configure is pending', async (_label, abandon) => {
    const base = mocks.request.getMockImplementation()!
    let finishConfigure!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) =>
      method === 'profiles.describe'
        ? Promise.resolve({ mcp_servers: [], skills: [{ enabled: true, name: 'arxiv' }], toolsets: [] })
        : method === 'profiles.configure'
          ? new Promise(resolve => {
              finishConfigure = () => resolve({})
            })
          : base(method, params)
    )

    // The staged tabs are where a capability pick (and so a configure) is made
    // before the profile exists, and where MCP setup mints the draft.
    await renderDialog(false)
    fireEvent.click(screen.getByRole('button', { name: 'Skills' }))
    fireEvent.click(await screen.findByRole('checkbox', { name: 'arxiv' }))
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('profiles.configure', expect.anything()))

    abandon()
    finishConfigure()
    await new Promise(resolve => setTimeout(resolve, 50))

    expect(mocks.saveBotMeta).not.toHaveBeenCalled()
  })
})

describe('provider readiness before the intro turn', () => {
  const notReady = { error: 'No usable credentials found for openrouter.', ok: false }

  it('keeps the bot, skips the intro, and offers model setup when its provider is not ready', async () => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) =>
      method === 'setup.runtime_check' ? notReady : base(method, params)
    )
    const onConfigureModel = vi.fn()

    await renderDialog(true, onConfigureModel)
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: false }))
    expect(mocks.request).toHaveBeenCalledWith('setup.runtime_check', { profile: 'inbox-triage' })
    expect(createCalls()).toHaveLength(1)

    const toast = mocks.notify.mock.calls.at(-1)![0]

    expect(toast).toMatchObject({
      detail: notReady.error,
      kind: 'warning',
      title: 'Bot "Inbox Triage" created'
    })
    toast.action.onClick()
    expect(onConfigureModel).toHaveBeenCalledWith(expect.objectContaining({ name: 'inbox-triage' }))
  })

  it('runs the intro when the provider check passes', async () => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) =>
      method === 'setup.runtime_check' ? { ok: true } : base(method, params)
    )

    await renderDialog(true)
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(mocks.notify).toHaveBeenLastCalledWith({ kind: 'success', message: 'Bot "Inbox Triage" created' })
  })

  it('checks a remote bot on its own machine and warns without an editor shortcut', async () => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])
    mocks.requestProfile.mockImplementation(async () => notReady)

    await renderDialog(true, vi.fn())
    await screen.findByText('Create on')
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    await waitFor(() =>
      expect(mocks.notify).toHaveBeenLastCalledWith(
        expect.objectContaining({ kind: 'warning', title: 'Bot "Inbox Triage" created on Studio' })
      )
    )
    expect(mocks.requestProfile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'studio' }),
      'setup.runtime_check',
      { profile: 'inbox-triage' }
    )
    expect(mocks.notify.mock.calls.at(-1)![0].action).toBeUndefined()
    expect(mocks.request).not.toHaveBeenCalledWith('setup.runtime_check', expect.anything())
  })
})

describe('profile creation warnings', () => {
  const warnings = ['Subscription logins stay with default.', 'A plugin could not be copied.']

  async function showNotifications() {
    const { NotificationStack } = await import('@/components/notifications')
    const { clearNotifications, notify } = await import('@/store/notifications')
    const { $notifications, dismissNotification } = await import('@/store/notifications')

    for (const notice of $notifications.get()) {
      dismissNotification(notice.id)
    }

    clearNotifications()
    mocks.notify.mockImplementation(notify)

    return render(<NotificationStack />)
  }

  it('joins lazy configure before adopting creation warnings', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishConfigure!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => {
      if (method === 'profiles.create') {
        return Promise.resolve({ warnings })
      }

      if (method === 'profiles.describe') {
        return Promise.resolve({ skills: [{ name: 'arxiv', enabled: true }] })
      }

      if (method === 'profiles.configure') {
        return new Promise(resolve => { finishConfigure = () => resolve({}) })
      }

      return base(method, params)
    })
    await renderDialog(false)
    fireEvent.click(screen.getByRole('button', { name: 'Skills' }))
    fireEvent.click(await screen.findByRole('checkbox', { name: 'arxiv' }))
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    await waitFor(() => expect(finishConfigure).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await act(async () => { await Promise.resolve() })
    expect(mocks.createCanonicalChat).not.toHaveBeenCalled()
    expect(mocks.request.mock.calls.filter(([method]) => method === 'setup.runtime_check')).toHaveLength(0)
    await act(async () => finishConfigure())
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.message === warnings.join('\n'))).toEqual([
      [expect.objectContaining({ title: 'inbox-triage — local', persistent: true })]
    ])
  })

  it('retires a pending named draft before creating its replacement', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishOld!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => {
      if (method === 'profiles.create') {
        return (params as { name: string }).name === 'inbox-triage'
          ? new Promise(resolve => { finishOld = () => resolve({ warnings: ['Old warning'] }) })
          : Promise.resolve({ warnings })
      }

      return base(method, params)
    })
    await renderDialog(true)
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(finishOld).toBeTypeOf('function'))
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'replacement' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await act(async () => finishOld())
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('replacement', { kickoff: true }))
    expect(mocks.deleteBot).toHaveBeenCalledWith({ name: 'inbox-triage' })
    expect(createCalls().map(([, params]) => params.name)).toEqual(['inbox-triage', 'replacement'])
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toEqual([
      [expect.objectContaining({ message: warnings.join('\n'), title: 'replacement — local' })]
    ])
    expect(mocks.saveBotMeta).toHaveBeenCalledWith('replacement', expect.anything())
  })

  it('disables name and target controls during final readiness', async () => {
    mocks.connections.mockResolvedValue([{ id: 'local', label: 'This Mac' }, { id: 'studio', label: 'Studio' }])
    const base = mocks.request.getMockImplementation()!
    let finishReadiness!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'setup.runtime_check'
      ? new Promise(resolve => { finishReadiness = () => resolve({ ok: true }) })
      : base(method, params)
    )
    await renderDialog(true)
    await screen.findByText('Create on')
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(finishReadiness).toBeTypeOf('function'))
    const name = screen.getByPlaceholderText('inbox-triage') as HTMLInputElement
    const target = controlUnder('Create on') as HTMLButtonElement
    expect(name.disabled).toBe(true)
    expect(target.disabled).toBe(true)
    // Native click honors disabled controls; fireEvent would bypass that policy.
    await act(async () => target.click())
    expect(screen.queryByRole('option', { name: 'Studio' })).toBeNull()
    await act(async () => finishReadiness())
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(mocks.requestProfile).not.toHaveBeenCalled()
    expect(mocks.deleteBot).not.toHaveBeenCalled()
  })

  it.each(['pending', 'ready'])('shows warnings once for a %s lazy create and keeps them after the dialog closes', async state => {
    const base = mocks.request.getMockImplementation()!
    let finishCreate!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) =>
      method === 'profiles.create'
        ? new Promise(resolve => { finishCreate = () => resolve({ warnings }) })
        : base(method, params)
    )
    const dialog = await renderDialog(true)
    await showNotifications()
    const { clearResumeToasts } = await import('@/app/session/hooks/use-session-actions/resume-latches')
    mocks.createCanonicalChat.mockImplementationOnce(async () => {
      clearResumeToasts(undefined)

      return 'session-1'
    })
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(createCalls()).toHaveLength(1))
    expect(screen.queryByText(/Subscription logins/)).toBeNull()

    if (state === 'ready') {
      finishCreate()
      await waitFor(() => expect(mocks.skillsView.at(-1)).toMatchObject({ fixedProfile: 'inbox-triage' }))
      expect(screen.queryByText(/Subscription logins/)).toBeNull()
    }

    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))

    if (state === 'pending') {
      finishCreate()
    }

    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    dialog.unmount()
    // Route-driven resume may clear again after the canonical opener returns.
    await act(async () => clearResumeToasts(undefined))
    const { $notifications } = await import('@/store/notifications')
    expect($notifications.get().filter(notice => notice.message === warnings.join('\n'))).toHaveLength(1)

    for (const warning of warnings) {
      expect(screen.getByText(text => text.includes(warning))).toBeTruthy()
    }

    expect(createCalls()).toHaveLength(1)
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toEqual([
      [expect.objectContaining({ title: 'inbox-triage — local', message: warnings.join('\n'), persistent: true })]
    ])
    expect(mocks.notify).toHaveBeenLastCalledWith({ kind: 'success', message: 'Bot "Inbox Triage" created' })
  })

  it.each(['cancel', 'target', 'name'])('retires completed draft warnings on %s and never adopts them into a replacement', async replacement => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) =>
      method === 'profiles.create' ? { warnings } : base(method, params)
    )
    await renderDialog(true)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(mocks.skillsView.at(-1)).toMatchObject({ fixedProfile: 'inbox-triage' }))
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toHaveLength(0)
    mocks.request.mockImplementation(base)

    if (replacement === 'target') {
      fireEvent.click(controlUnder('Create on'))
      fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    } else if (replacement === 'cancel') {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    }

    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'replacement' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', message: expect.stringContaining('Replacement') })))
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toHaveLength(0)
  })

  it('attributes an adopted remote warning to the captured profile and machine', async () => {
    mocks.connections.mockResolvedValue([
      { id: 'local', label: 'This Mac' },
      { id: 'studio', label: 'Studio' }
    ])
    mocks.requestProfile.mockImplementation(async (_route, method) => method === 'profiles.create' ? { warnings } : {})
    await renderDialog(true)
    await showNotifications()
    await screen.findByText('Create on')
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(mocks.skillsView.at(-1)).toMatchObject({ fixedConnection: 'studio', fixedProfile: 'inbox-triage' }))
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toHaveLength(0)
    fireEvent.change(screen.getByPlaceholderText('Inbox Triage'), { target: { value: 'Changed title' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.notify).toHaveBeenCalledWith({
      kind: 'warning', title: 'inbox-triage — Studio (studio)', message: warnings.join('\n'), persistent: true
    }))
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toHaveLength(1)
  })

  it('does not adopt warnings after unmount during readiness', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishReadiness!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => {
      if (method === 'profiles.create') {
        return Promise.resolve({ warnings })
      }

      if (method === 'setup.runtime_check') {
        return new Promise(resolve => { finishReadiness = () => resolve({ ok: true }) })
      }

      return base(method, params)
    })
    const dialog = await renderDialog(true)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(finishReadiness).toBeTypeOf('function'))

    dialog.unmount()
    await act(async () => finishReadiness())
    expect(mocks.notify).not.toHaveBeenCalled()
    expect(mocks.createCanonicalChat).not.toHaveBeenCalled()
  })

  it('keeps the model setup notice and action available when creation also returned warnings', async () => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) =>
      method === 'profiles.create' ? { warnings }
        : method === 'setup.runtime_check' ? { ok: false, error: 'No usable credentials.' }
          : base(method, params)
    )
    const configure = vi.fn()
    const dialog = await renderDialog(true, configure)
    await showNotifications()
    const { clearResumeToasts } = await import('@/app/session/hooks/use-session-actions/resume-latches')
    mocks.createCanonicalChat.mockImplementationOnce(async () => {
      clearResumeToasts(undefined)

      return 'session-1'
    })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: false }))
    dialog.unmount()
    await act(async () => clearResumeToasts(undefined))
    // The existing stack retains its front card; expanding exposes both notices.
    fireEvent.click(screen.getByRole('button', { name: /Show.*more/ }))
    expect(screen.getByText(text => text.includes(warnings[0]))).toBeTruthy()
    fireEvent.click(await screen.findByRole('button', { name: 'Configure model' }))
    expect(configure).toHaveBeenCalledWith(expect.objectContaining({ name: 'inbox-triage' }))
    expect(mocks.notify).toHaveBeenLastCalledWith(expect.objectContaining({
      kind: 'warning', title: 'Bot "Inbox Triage" created', detail: 'No usable credentials.'
    }))
  })

  it('explicit Clear all dismisses persistent creation and needs-model cards', async () => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) => method === 'profiles.create'
      ? { warnings } : method === 'setup.runtime_check' ? { ok: false, error: 'No credentials.' } : base(method, params))
    const dialog = await renderDialog(true, vi.fn())
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    dialog.unmount()
    const { clearNotifications, $notifications } = await import('@/store/notifications')
    await act(async () => clearNotifications())
    expect($notifications.get().filter(notice => notice.persistent)).toHaveLength(2)
    fireEvent.click(screen.getByRole('button', { name: /Show.*more/ }))
    expect(screen.getByText(text => text.includes(warnings[0]))).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Configure model' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }))
    expect($notifications.get()).toEqual([])
    await waitFor(() => expect(screen.queryByText(text => text.includes(warnings[0]))).toBeNull())
    expect(screen.queryByRole('button', { name: 'Configure model' })).toBeNull()
  })

  it.each([{}, { warnings: [] }])('keeps successful creation quiet for a warning-free reply %j', async reply => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) =>
      method === 'profiles.create' ? reply : base(method, params)
    )
    await renderDialog(true)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    expect(screen.getByText('Bot "Inbox Triage" created')).toBeTruthy()
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toHaveLength(0)
  })

  it.each(['cancel', 'unmount'])('suppresses warnings when abandoned during configure (%s)', async abandon => {
    const base = mocks.request.getMockImplementation()!
    let finishConfigure!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => {
      if (method === 'profiles.create') {
        return Promise.resolve({ warnings })
      }

      if (method === 'profiles.describe') {
        return Promise.resolve({ mcp_servers: [], skills: [{ enabled: true, name: 'arxiv' }], toolsets: [] })
      }

      if (method === 'profiles.configure') {
        return new Promise(resolve => { finishConfigure = () => resolve({}) })
      }

      return base(method, params)
    })
    const dialog = await renderDialog(false)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Skills' }))
    fireEvent.click(await screen.findByRole('checkbox', { name: 'arxiv' }))
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('profiles.configure', expect.anything()))
    expect(screen.queryByText(text => text.includes(warnings[0]))).toBeNull()

    if (abandon === 'cancel') {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    } else {
      dialog.unmount()
    }

    await act(async () => { finishConfigure() })
    expect(mocks.saveBotMeta).not.toHaveBeenCalled()
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toHaveLength(0)
    expect(screen.queryByText(text => text.includes(warnings[0]))).toBeNull()
  })

  it('does not notify for an abandoned flight or leak its warnings into a reused dialog', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishCreate!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) =>
      method === 'profiles.create'
        ? new Promise(resolve => { finishCreate = () => resolve({ warnings }) })
        : base(method, params)
    )
    await renderDialog(true)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(createCalls()).toHaveLength(1))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    finishCreate()
    await waitFor(() => expect(mocks.deleteBot).toHaveBeenCalled())
    expect(screen.queryByText(text => text.includes(warnings[0]))).toBeNull()
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toHaveLength(0)

    mocks.request.mockImplementation(base)
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'second' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('second', { kickoff: true }))
    expect(screen.queryByText(text => text.includes(warnings[0]))).toBeNull()
  })
})

describe('retired staged MCP owners', () => {
  const rename = () => fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'replacement' } })

  async function openSetup() {
    const dialog = await renderDialog(false)
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    fireEvent.click(await screen.findByRole('button', { name: /Set up|Sign in/ }))

    return dialog
  }

  it('drops the real key form and cached profile when renamed', async () => {
    await openSetup()
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'old-key' } })
    rename()
    expect(screen.queryByPlaceholderText('LINEAR_API_KEY')).toBeNull()
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    const key = await screen.findByPlaceholderText('LINEAR_API_KEY') as HTMLInputElement
    expect(key.value).toBe('')
    fireEvent.change(key, { target: { value: 'new-key' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.test', { name: 'linear', profile: 'replacement' }))
    expect(mocks.request.mock.calls.filter(([method]) => method === 'mcp.servers.set_api_key')).toEqual([
      ['mcp.servers.set_api_key', { name: 'linear', profile: 'replacement', env_var: 'LINEAR_API_KEY', value: 'new-key' }]
    ])
  })

  it.each(['profiles.create', 'mcp.servers.add', 'mcp.servers.set_api_key', 'mcp.servers.test'].flatMap(pending =>
    ['rename', 'unmount'].map(retire => [pending, retire])
  ))(
    'stops subsequent operations and publication during %s on %s', async (pending, retire) => {
      const base = mocks.request.getMockImplementation()!
      let finish!: () => void
      mocks.request.mockImplementation((method: string, params?: unknown) => {
        if (method === 'mcp.catalog') {
          return Promise.resolve({ servers: [{ ...catalog, requires: ['LINEAR_API_KEY', 'SECOND_KEY'] }] })
        }

        if (method === pending) {
          return new Promise(resolve => { finish = () => resolve({ ok: true }) })
        }

        return base(method, params)
      })
      const dialog = await openSetup()

      if (pending.includes('set_api_key') || pending.includes('.test')) {
        fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'first' } })
        fireEvent.change(screen.getByPlaceholderText('SECOND_KEY'), { target: { value: 'second' } })
        fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
      }

      await waitFor(() => expect(finish).toBeTypeOf('function'))

      if (retire === 'rename') {
        rename()
      } else {
        dialog.unmount()
      }

      const callsAtRetirement = mocks.request.mock.calls.length
      await act(async () => finish())
      expect(mocks.request.mock.calls.slice(callsAtRetirement).filter(([method]) => method.startsWith('mcp.servers.'))).toEqual([])
      expect(mocks.notify.mock.calls.filter(([notice]) => /configured|authenticated/i.test(notice.message))).toEqual([])

      if (retire === 'rename') {
        expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('unchecked')
      }
    }
  )

  it.each(['rename', 'unmount'])('retires pending OAuth on %s without updating replacement capabilities', async retire => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'mcp.catalog'
      ? Promise.resolve({ servers: [{ ...catalog, auth: 'oauth' }] }) : base(method, params))
    let finish!: () => void
    mocks.completeMcpOAuth.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve }))
    const dialog = await openSetup()
    await waitFor(() => expect(finish).toBeTypeOf('function'))
    const options = mocks.completeMcpOAuth.mock.calls.at(-1)![0]
    expect(options.cancelled()).toBe(false)

    if (retire === 'rename') {
      rename()
    } else {
      dialog.unmount()
    }

    expect(options.cancelled()).toBe(true)
    await act(async () => finish())
    expect(mocks.notify.mock.calls.filter(([notice]) => /authenticated/i.test(notice.message))).toEqual([])

    if (retire === 'rename') {
      expect(screen.getByRole('button', { name: /Sign in/ })).toBeTruthy()
      expect((screen.getByRole('checkbox', { name: /linear/ }) as HTMLButtonElement).getAttribute('data-state')).toBe('unchecked')
    }
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

    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())

    // Second open: the picker must read `default` again, or every agent after
    // the first silently starts from a bare profile.
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }))
    expect(cloneFrom().textContent).toBe('default')
  })
})
