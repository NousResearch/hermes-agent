/**
 * New Bot MCP setup against a lazily created draft: renaming retires pending
 * setup flows, replacing the draft drops its completed setup, and adopting the
 * bot never cuts off setup that is still writing keys.
 */

import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { catalog, createAgentHarness } from './create-agent-test-utils'
import { translateBots } from './i18n-test-helper'

const mocks = await vi.hoisted(async () => (await import('./create-agent-test-utils')).createMocks())

vi.mock('@hermes/plugin-sdk', async importOriginal =>
  (await import('./create-agent-test-utils')).sdkMock(await importOriginal(), mocks, () => translateBots)
)
vi.mock('./canonical-chat', () => ({ createCanonicalChat: mocks.createCanonicalChat }))
vi.mock('./profile-ops', () => ({ deleteBot: mocks.deleteBot }))
vi.mock('./data', async importOriginal => ({ ...(await importOriginal<object>()), saveBotMeta: mocks.saveBotMeta }))

const { renderDialog } = createAgentHarness(mocks)

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

  it.each(['key', 'oauth'])('retires completed %s setup on rename without dropping source installations', async auth => {
    const base = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async (method: string, params?: unknown) => method === 'mcp.catalog'
      ? { servers: [{ ...catalog, auth: auth === 'oauth' ? 'oauth' : null }] }
      : method === 'profiles.describe'
        ? { mcp_servers: [{ name: 'source-server', enabled: true, installed: true }], skills: [], toolsets: [] }
        : base(method, params))
    mocks.completeMcpOAuth.mockResolvedValue(undefined)
    await openSetup()

    if (auth === 'key') {
      fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'old-draft-key' } })
      fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    }

    await waitFor(() => expect(screen.queryByRole('button', { name: /Set up|Sign in/ })).toBeNull())
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('checked')

    rename()
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('unchecked')
    expect(screen.getByRole('checkbox', { name: /linear/ }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('checkbox', { name: /source-server/ }).getAttribute('data-state')).toBe('checked')
    fireEvent.click(await screen.findByRole('button', { name: /Set up|Sign in/ }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('profiles.configure', {
      name: 'replacement', enabled_mcp_servers: ['source-server']
    }))

    if (auth === 'key') {
      const key = await screen.findByPlaceholderText('LINEAR_API_KEY') as HTMLInputElement
      expect(key.value).toBe('')
      fireEvent.change(key, { target: { value: 'replacement-key' } })
      fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
      await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.set_api_key', {
        name: 'linear', profile: 'replacement', env_var: 'LINEAR_API_KEY', value: 'replacement-key'
      }))
      expect(mocks.request.mock.calls.filter(([method, params]) => method === 'mcp.servers.set_api_key'
        && params.profile === 'replacement' && params.value === 'old-draft-key')).toEqual([])
    } else {
      await waitFor(() => expect(mocks.completeMcpOAuth).toHaveBeenCalledTimes(2))
      expect(mocks.completeMcpOAuth.mock.calls.at(-1)![0]).toMatchObject({ profile: { profile: 'replacement' } })
    }
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

  it('finishes key setup that is still writing when the bot is created', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishFirst!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => {
      if (method === 'mcp.catalog') {
        return Promise.resolve({ servers: [{ ...catalog, requires: ['LINEAR_API_KEY', 'SECOND_KEY'] }] })
      }

      if (method === 'mcp.servers.set_api_key' && (params as { env_var: string }).env_var === 'LINEAR_API_KEY') {
        return new Promise(resolve => { finishFirst = () => resolve({ ok: true }) })
      }

      return base(method, params)
    })
    await openSetup()
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'first' } })
    fireEvent.change(screen.getByPlaceholderText('SECOND_KEY'), { target: { value: 'second' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(finishFirst).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    await act(async () => finishFirst())
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.test', { name: 'linear', profile: 'inbox-triage' }))
    expect(mocks.request).toHaveBeenCalledWith('mcp.servers.set_api_key', {
      name: 'linear', profile: 'inbox-triage', env_var: 'SECOND_KEY', value: 'second'
    })
  })

  it('keeps completed setup on a draft renamed away and back', async () => {
    await openSetup()
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'kept-key' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /Set up/ })).toBeNull())
    rename()
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('unchecked')
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    expect(screen.queryByRole('button', { name: /Set up/ })).toBeNull()
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('checked')
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    expect(mocks.request.mock.calls.filter(([method]) => method === 'mcp.servers.add')).toHaveLength(1)
    expect(mocks.deleteBot).not.toHaveBeenCalled()
  })

  it('resumes an added server at its keys after a rename away and back', async () => {
    await openSetup()
    await screen.findByPlaceholderText('LINEAR_API_KEY')
    expect(mocks.request.mock.calls.filter(([method]) => method === 'mcp.servers.add')).toHaveLength(1)
    rename()
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'resumed' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.test', { name: 'linear', profile: 'inbox-triage' }))
    expect(mocks.request.mock.calls.filter(([method]) => method === 'mcp.servers.add')).toHaveLength(1)
  })

  it('reports a key setup that fails after the bot was created, without touching the next dialog', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishTest!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'mcp.servers.test'
      ? new Promise(resolve => { finishTest = () => resolve({ ok: false, error: 'bad key' }) })
      : base(method, params))
    await openSetup()
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'bad' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(finishTest).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    await act(async () => finishTest())
    expect(mocks.notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', title: 'linear', message: 'bad key', persistent: true }))
  })

  it('does not mark the reopened dialog when an adopted setup finishes later', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishTest!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'mcp.servers.test'
      ? new Promise(resolve => { finishTest = () => resolve({ ok: true }) })
      : base(method, params))
    await openSetup()
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'slow' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(finishTest).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }))
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    await screen.findByRole('button', { name: /Set up/ })
    await act(async () => finishTest())
    expect(mocks.notify).toHaveBeenCalledWith({ kind: 'success', message: 'linear configured' })
    expect(screen.getByRole('button', { name: /Set up/ })).toBeTruthy()
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('unchecked')
  })

  it('lets an earlier bot finish its key setup after a second bot is created', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishFirst!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => {
      if (method === 'mcp.catalog') {
        return Promise.resolve({ servers: [{ ...catalog, requires: ['LINEAR_API_KEY', 'SECOND_KEY'] }] })
      }

      const p = params as { env_var?: string; profile?: string }

      if (method === 'mcp.servers.set_api_key' && p.profile === 'inbox-triage' && p.env_var === 'LINEAR_API_KEY') {
        return new Promise(resolve => { finishFirst = () => resolve({ ok: true }) })
      }

      return base(method, params)
    })
    await openSetup()
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'first' } })
    fireEvent.change(screen.getByPlaceholderText('SECOND_KEY'), { target: { value: 'second' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(finishFirst).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'second-bot' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('second-bot', { kickoff: true }))
    await act(async () => finishFirst())
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.test', { name: 'linear', profile: 'inbox-triage' }))
    expect(mocks.request).toHaveBeenCalledWith('mcp.servers.set_api_key', {
      name: 'linear', profile: 'inbox-triage', env_var: 'SECOND_KEY', value: 'second'
    })
  })

  it('records an add that finishes while the draft is renamed away', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishAdd!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'mcp.servers.add' && !finishAdd
      ? new Promise(resolve => { finishAdd = () => resolve({ ok: true }) })
      : base(method, params))
    await openSetup()
    await waitFor(() => expect(finishAdd).toBeTypeOf('function'))
    rename()
    await act(async () => finishAdd())
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'resumed' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.test', { name: 'linear', profile: 'inbox-triage' }))
    expect(mocks.request.mock.calls.filter(([method]) => method === 'mcp.servers.add')).toHaveLength(1)
  })

  it('keeps a setup that finishes while another tab is open', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishTest!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'mcp.servers.test'
      ? new Promise(resolve => { finishTest = () => resolve({ ok: true }) })
      : base(method, params))
    await openSetup()
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'k' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(finishTest).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: 'General' }))
    await act(async () => finishTest())
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    expect(screen.queryByRole('button', { name: /Set up/ })).toBeNull()
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('checked')
  })

  it('records setup started on a renamed name, whose click replaced the old draft', async () => {
    await openSetup()
    await screen.findByPlaceholderText('LINEAR_API_KEY')
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'other-bot' } })
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'k' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.test', { name: 'linear', profile: 'other-bot' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /Set up/ })).toBeNull())
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('checked')
  })

  it('does not mark the reopened dialog when an adopted add finishes later', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishAdd!: () => void
    mocks.request.mockImplementation((method: string, params?: unknown) => method === 'mcp.servers.add' && !finishAdd
      ? new Promise(resolve => { finishAdd = () => resolve({ ok: true }) })
      : base(method, params))
    await openSetup()
    await waitFor(() => expect(finishAdd).toBeTypeOf('function'))
    fireEvent.click(screen.getByRole('button', { name: 'Create bot' }))
    await waitFor(() => expect(mocks.createCanonicalChat).toHaveBeenCalledWith('inbox-triage', { kickoff: true }))
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }))
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    await act(async () => finishAdd())
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    await screen.findByPlaceholderText('LINEAR_API_KEY')
    expect(mocks.request.mock.calls.filter(([method]) => method === 'mcp.servers.add')).toHaveLength(2)
  })

  it('drops a late add for a draft that was replaced and recreated under the same name', async () => {
    const base = mocks.request.getMockImplementation()!
    let finishAdd!: () => void
    let triageAdds = 0

    const adds = () => mocks.request.mock.calls.filter(([method, p]) =>
      method === 'mcp.servers.add' && (p as { profile: string }).profile === 'inbox-triage')

    mocks.request.mockImplementation((method: string, params?: unknown) => {
      if (method === 'mcp.servers.add' && (params as { profile: string }).profile === 'inbox-triage') {
        triageAdds += 1

        // The first draft's add is held; the recreated draft's own add fails.
        return triageAdds === 1
          ? new Promise(resolve => { finishAdd = () => resolve({ ok: true }) })
          : Promise.reject(new Error('add failed'))
      }

      return base(method, params)
    })
    await openSetup()
    await waitFor(() => expect(finishAdd).toBeTypeOf('function'))
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'other-bot' } })
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    await screen.findByPlaceholderText('LINEAR_API_KEY')
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    await waitFor(() => expect(adds()).toHaveLength(2))
    await act(async () => finishAdd())
    rename()
    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    // The recreated draft never got the server, so its setup must add it.
    await waitFor(() => expect(adds()).toHaveLength(3))
  })

  it('completes setup under StrictMode, as the app root renders it', async () => {
    await renderDialog(false, undefined, true)
    fireEvent.click(screen.getByRole('button', { name: 'MCP' }))
    fireEvent.click(await screen.findByRole('button', { name: /Set up/ }))
    fireEvent.change(await screen.findByPlaceholderText('LINEAR_API_KEY'), { target: { value: 'strict' } })
    fireEvent.click(screen.getByRole('button', { name: /Save.*test/i }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith('mcp.servers.test', { name: 'linear', profile: 'inbox-triage' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /Set up/ })).toBeNull())
    expect(screen.getByRole('checkbox', { name: /linear/ }).getAttribute('data-state')).toBe('checked')
  })
})
