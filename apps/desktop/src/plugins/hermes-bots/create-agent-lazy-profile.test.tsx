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

import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { controlUnder, createAgentHarness } from './create-agent-test-utils'
import { translateBots } from './i18n-test-helper'

const mocks = await vi.hoisted(async () => (await import('./create-agent-test-utils')).createMocks())

vi.mock('@hermes/plugin-sdk', async importOriginal =>
  (await import('./create-agent-test-utils')).sdkMock(await importOriginal(), mocks, () => translateBots)
)
vi.mock('./canonical-chat', () => ({ createCanonicalChat: mocks.createCanonicalChat }))
vi.mock('./profile-ops', () => ({ deleteBot: mocks.deleteBot }))
vi.mock('./data', async importOriginal => ({ ...(await importOriginal<object>()), saveBotMeta: mocks.saveBotMeta }))

const { createCalls, renderDialog } = createAgentHarness(mocks)

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
