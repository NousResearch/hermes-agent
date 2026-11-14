/**
 * New Bot creation warnings: a lazily created draft owns its `profiles.create`
 * warnings until the bot is adopted, then publishes them once as a persistent
 * notice that outlives the dialog, the Bot Chat opening and resume clears.
 */

import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { controlUnder, createAgentHarness, dismissAllNotifications } from './create-agent-test-utils'
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
    const { notify } = await import('@/store/notifications')
    await dismissAllNotifications()
    mocks.notify.mockImplementation(notify)

    return render(<NotificationStack />)
  }

  it('publishes the warnings of a draft still applying its picks when Create is clicked', async () => {
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
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    await act(async () => finishConfigure())
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.message === warnings.join('\n'))).toEqual([
      [expect.objectContaining({ title: 'inbox-triage — local', persistent: true })]
    ])
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

  it('opens Configure model on the live roster row of the creating connection, after a gateway switch', async () => {
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
    mocks.activeConnection.value = 'gateway-b'
    const { $lastRoster } = await import('./data')

    const live: RosterRow = {
      name: 'inbox-triage', connectionId: 'gateway-a', sourceScoped: true,
      route: { connectionId: 'gateway-a', mode: 'remote', profile: 'inbox-triage', targetProfile: 'inbox-triage' }
    }

    const previous = $lastRoster.get()
    // A same-named bot on the gateway switched to must not be picked.
    $lastRoster.set([{ ...live, connectionId: 'gateway-b', route: { ...live.route!, connectionId: 'gateway-b' } }, live])

    try {
      const { clearResumeToasts } = await import('@/app/session/hooks/use-session-actions/resume-latches')
      await act(async () => clearResumeToasts(undefined))
      fireEvent.click(await screen.findByRole('button', { name: 'Configure model' }))
      expect(configure).toHaveBeenCalledWith(live)
    } finally {
      $lastRoster.set(previous)
    }
  })

  it('captures the gateway at delayed draft creation without a render after switching', async () => {
    mocks.activeConnection.value = 'gateway-a'
    mocks.connections.mockResolvedValue([
      { id: 'gateway-a', label: 'Machine A' },
      { id: 'gateway-b', label: 'Machine B' }
    ])
    const base = mocks.request.getMockImplementation()!
    const createdOn: string[] = []
    let finishCreate!: () => void
    mocks.request.mockImplementation(async (method: string, params?: unknown) => {
      if (method === 'profiles.create') {
        createdOn.push(mocks.activeConnection.value)

        return new Promise(resolve => { finishCreate = () => resolve({ warnings }) })
      }

      return method === 'setup.runtime_check' ? { ok: false, error: 'No credentials.' } : base(method, params)
    })
    const configure = vi.fn()
    await renderDialog(true, configure)
    await showNotifications()
    await screen.findByText('Create on')
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: '' } })
    vi.useFakeTimers()

    try {
      fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
      fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
      expect(createCalls()).toHaveLength(0)
      // Changing the host state alone does not render the dialog. The pending
      // timer still holds the callback from the render on Machine A.
      mocks.activeConnection.value = 'gateway-b'
      await act(async () => vi.advanceTimersByTimeAsync(800))
    } finally {
      vi.useRealTimers()
    }

    expect(createdOn).toEqual(['gateway-b'])
    // Switching again while the RPC is pending must not change its owner.
    mocks.activeConnection.value = 'gateway-a'
    await act(async () => finishCreate())
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    expect.soft(mocks.notify).toHaveBeenCalledWith({
      kind: 'warning', title: 'inbox-triage — Machine B (gateway-b)',
      message: warnings.join('\n'), persistent: true
    })

    const { $lastRoster } = await import('./data')
    const previous = $lastRoster.get()
    const live: RosterRow = { name: 'inbox-triage', connectionId: 'gateway-b', sourceScoped: true }
    $lastRoster.set([{ ...live, connectionId: 'gateway-a' }, live])
    mocks.activeConnection.value = 'gateway-a'

    try {
      fireEvent.click(screen.getByRole('button', { name: /Show.*more/ }))
      fireEvent.click(await screen.findByRole('button', { name: 'Configure model' }))
      expect(configure).toHaveBeenCalledWith(live)
    } finally {
      $lastRoster.set(previous)
    }
  })

  it('pins Configure model to the creating connection when the roster does not list the bot after a switch', async () => {
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
    mocks.activeConnection.value = 'gateway-b'
    const { clearResumeToasts } = await import('@/app/session/hooks/use-session-actions/resume-latches')
    await act(async () => clearResumeToasts(undefined))
    fireEvent.click(await screen.findByRole('button', { name: 'Configure model' }))
    expect(configure).toHaveBeenCalledWith(expect.objectContaining({ name: 'inbox-triage', connectionId: 'gateway-a', sourceScoped: true }))
  })

  it('drops the warnings of a bot a target change discarded during readiness', async () => {
    mocks.connections.mockResolvedValue([{ id: 'local', label: 'This Mac' }, { id: 'studio', label: 'Studio' }])
    const base = mocks.request.getMockImplementation()!
    let finishReadiness!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'profiles.create'
      ? Promise.resolve({ warnings })
      : method === 'setup.runtime_check'
        ? new Promise(resolve => { finishReadiness = () => resolve({ ok: true }) })
        : base(method, params)
    )
    await renderDialog(true)
    await screen.findByText('Create on')
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(finishReadiness).toBeTypeOf('function'))
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    await act(async () => finishReadiness())
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.message === warnings.join('\n'))).toEqual([])
  })

  it('drops the warnings of a bot replaced under the same name during readiness', async () => {
    mocks.connections.mockResolvedValue([{ id: 'local', label: 'This Mac' }, { id: 'studio', label: 'Studio' }])
    const base = mocks.request.getMockImplementation()!
    let finishReadiness!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'profiles.create'
      ? Promise.resolve({ warnings })
      : method === 'setup.runtime_check'
        ? new Promise(resolve => { finishReadiness = () => resolve({ ok: true }) })
        : base(method, params)
    )
    await renderDialog(true)
    await screen.findByText('Create on')
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(finishReadiness).toBeTypeOf('function'))
    fireEvent.click(controlUnder('Create on'))
    fireEvent.click(await screen.findByRole('option', { name: 'Studio' }))
    // A same-named replacement draft materializes on the new target.
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(mocks.requestProfile.mock.calls.filter(([, method]) => method === 'profiles.create')).toHaveLength(1))
    await act(async () => finishReadiness())
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    expect(mocks.notify.mock.calls.filter(([notice]) => notice.message === warnings.join('\n'))).toEqual([])
  })

  it('pins Configure model to the connection the draft was created on, not the one active at Create', async () => {
    mocks.activeConnection.value = 'gateway-a'
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) => method === 'setup.runtime_check'
      ? { ok: false, error: 'No credentials.' } : base(method, params))
    const configure = vi.fn()
    const dialog = await renderDialog(true, configure)
    await showNotifications()
    fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
    await waitFor(() => expect(createCalls()).toHaveLength(1))
    mocks.activeConnection.value = 'gateway-b'
    // The switch re-renders the dialog (here: any edit), so Create reads gateway-b as active.
    fireEvent.click(screen.getByRole('button', { name: 'General' }))
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage ' } })
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalled())
    dialog.unmount()
    const { clearResumeToasts } = await import('@/app/session/hooks/use-session-actions/resume-latches')
    await act(async () => clearResumeToasts(undefined))
    fireEvent.click(await screen.findByRole('button', { name: 'Configure model' }))
    expect(configure).toHaveBeenCalledWith(expect.objectContaining({ name: 'inbox-triage', connectionId: 'gateway-a', sourceScoped: true }))
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
