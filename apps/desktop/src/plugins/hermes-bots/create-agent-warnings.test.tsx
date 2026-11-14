/**
 * New Bot creation warnings: a lazily created draft owns its `profiles.create`
 * warnings until the bot is adopted, then publishes them once as a persistent
 * notice that outlives the dialog, the Bot Chat opening and resume clears.
 */

import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { controlUnder, createAgentHarness } from './create-agent-test-utils'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const mocks = await vi.hoisted(async () => (await import('./create-agent-test-utils')).createMocks())

vi.mock('@hermes/plugin-sdk', async importOriginal =>
  (await import('./create-agent-test-utils')).sdkMock(await importOriginal(), mocks, () => translateBots)
)
vi.mock('./canonical-chat', () => ({ createCanonicalChat: mocks.createCanonicalChat }))
vi.mock('./profile-ops', () => ({ deleteBot: mocks.deleteBot }))
vi.mock('./data', async importOriginal => ({ ...(await importOriginal<object>()), saveBotMeta: mocks.saveBotMeta }))

const { createCalls, renderDialog } = createAgentHarness(mocks)

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

  it('still publishes an adopted creation when the dialog unmounts during readiness', async () => {
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
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.kind === 'warning')).toEqual([
      [expect.objectContaining({ title: 'inbox-triage — local', message: warnings.join('\n'), persistent: true })]
    ])
    expect(mocks.notify).toHaveBeenLastCalledWith({ kind: 'success', message: 'Bot "Inbox Triage" created' })
  })

  it.each(['created', 'pending'])('keeps a %s draft when renamed away and back before Create', async state => {
    const base = mocks.request.getMockImplementation()!
    let finishCreate: (() => void) | undefined
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'profiles.create' && state === 'pending' && !finishCreate
      ? new Promise(resolve => { finishCreate = () => resolve({}) })
      : base(method, params))
    await renderDialog(true)
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(createCalls()).toHaveLength(1))
    const input = screen.getByPlaceholderText('inbox-triage')
    fireEvent.change(input, { target: { value: 'inbox-triagex' } })
    fireEvent.change(input, { target: { value: 'inbox-triage' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await act(async () => finishCreate?.())
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(createCalls()).toHaveLength(1)
    expect(mocks.deleteBot).not.toHaveBeenCalled()
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

  it('keeps the persistent model action on its creation connection after switching gateways', async () => {
    mocks.activeConnection.value = 'gateway-a'
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) => method === 'setup.runtime_check'
      ? { ok: false, error: 'No credentials.' } : base(method, params))

    const configure = vi.fn(async (bot: RosterRow) => {
      const { requestForBot } = await import('./routing')
      await requestForBot(bot, 'profiles.configure', { name: bot.name, model: 'picked-model' })
    })

    const dialog = await renderDialog(true, configure)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    dialog.unmount()
    mocks.activeConnection.value = 'gateway-b'
    const { clearResumeToasts } = await import('@/app/session/hooks/use-session-actions/resume-latches')
    await act(async () => clearResumeToasts(undefined))
    fireEvent.click(await screen.findByRole('button', { name: 'Configure model' }))
    await waitFor(() => expect(mocks.requestProfile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'gateway-a', profile: 'inbox-triage', targetProfile: 'inbox-triage' }),
      'profiles.configure', expect.objectContaining({ name: 'inbox-triage' }), undefined, { spawnPriority: 'foreground' }
    ))
    expect(mocks.request.mock.calls.filter(([method]) => method === 'profiles.configure')).toEqual([])
  })

  it('opens Configure model on the live roster row once the roster has the bot', async () => {
    mocks.activeConnection.value = 'gateway-a'
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) => method === 'setup.runtime_check'
      ? { ok: false, error: 'No credentials.' } : base(method, params))
    const configure = vi.fn()
    const dialog = await renderDialog(true, configure)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    dialog.unmount()
    const { $lastRoster } = await import('./data')

    const live: RosterRow = {
      name: 'inbox-triage', connectionId: 'gateway-a', sourceScoped: true,
      route: { connectionId: 'gateway-a', mode: 'remote', profile: 'inbox-triage', targetProfile: 'inbox-triage' }
    }

    const previous = $lastRoster.get()
    $lastRoster.set([live])

    try {
      const { clearResumeToasts } = await import('@/app/session/hooks/use-session-actions/resume-latches')
      await act(async () => clearResumeToasts(undefined))
      fireEvent.click(await screen.findByRole('button', { name: 'Configure model' }))
      expect(configure).toHaveBeenCalledWith(live)
    } finally {
      $lastRoster.set(previous)
    }
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
    const { notify } = await import('@/store/notifications')
    act(() => void notify({ kind: 'success', message: 'transient', placement: 'bottom-right' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }))
    // Clear all still clears the transient bottom-right toasts too.
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
