import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import type * as GatewayStore from '@/store/gateway'
import type * as NotificationStore from '@/store/notifications'
import type * as SessionStates from '@/store/session-states'

import { ModelPresets } from './model-presets'

const mocks = vi.hoisted(() => ({ request: vi.fn(), confirm: vi.fn(), api: vi.fn(), error: vi.fn() }))

const { session, owner } = await vi.hoisted(async () => {
  const { atom } = await import('nanostores')

  return {
    session: atom<string | null>('runtime-work'),
    owner: atom<{ connectionId: string; profile: string } | null>({ connectionId: 'local', profile: 'work' })
  }
})

vi.mock('@/store/session-states', async importOriginal => ({
  ...(await importOriginal<typeof SessionStates>()),
  $focusedRuntimeId: session,
  $sessionStates: owner,
  knownOwnerForSession: () => owner.get()
}))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: mocks.request })
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  activeGatewayConnectionId: () => 'local',
  activeGatewayProfileKey: () => 'work'
}))
vi.mock('@/store/confirm', () => ({ confirm: mocks.confirm }))
vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<typeof NotificationStore>()),
  notifyError: mocks.error
}))

let state = { names: ['quick'], active: 'quick' }
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
  queryClient.clear()
  vi.clearAllMocks()
  session.set('runtime-work')
  owner.set({ connectionId: 'local', profile: 'work' })
  state = { names: ['quick'], active: 'quick' }
  mocks.confirm.mockResolvedValue(true)
  mocks.request.mockResolvedValue({ output: 'Applied quick' })
  mocks.api.mockImplementation(async request => {
    if (!request.body) {
      return structuredClone(state)
    }

    const body = request.body

    if (body.action === 'save') {
      state = { names: [...state.names, body.name], active: body.name }
    }

    if (body.action === 'rename') {
      state = { names: [body.new_name], active: body.new_name }
    }

    if (body.action === 'delete') {
      state = { names: [], active: '' }
    }

    return structuredClone(state)
  })
  window.hermesDesktop = { ...window.hermesDesktop, api: mocks.api }
})
afterEach(cleanup)

function mount(scope = 'work') {
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelPresets scope={scope} />
    </QueryClientProvider>
  )
}

describe('Model route presets', () => {
  it('saves current server routes by name, renames and deletes without posting snapshots', async () => {
    mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Save current as…' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Preset name' }), { target: { value: 'new-set' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Save current as…' }).at(-1)!)
    await waitFor(() =>
      expect(mocks.api).toHaveBeenCalledWith(
        expect.objectContaining({ profile: 'work', body: { action: 'save', name: 'new-set' } })
      )
    )
    await waitFor(() => expect(screen.queryByRole('textbox')).toBeNull())
    fireEvent.click(screen.getByRole('button', { name: 'Rename preset' }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'renamed' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Rename preset' }).at(-1)!)
    await waitFor(() => expect(screen.queryByRole('textbox')).toBeNull())
    expect(mocks.api).toHaveBeenCalledWith(
      expect.objectContaining({ body: { action: 'rename', name: 'new-set', new_name: 'renamed' } })
    )
    fireEvent.click(screen.getByRole('button', { name: 'Delete preset' }))
    await waitFor(() =>
      expect(mocks.api).toHaveBeenCalledWith(expect.objectContaining({ body: { action: 'delete', name: 'renamed' } }))
    )
  })

  it('closes a name dialog rather than saving into a newly selected profile', async () => {
    const view = mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Save current as…' }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'do-not-retarget' } })
    view.rerender(
      <QueryClientProvider client={queryClient}>
        <ModelPresets scope="other" />
      </QueryClientProvider>
    )
    await waitFor(() => expect(screen.queryByRole('textbox')).toBeNull())
    expect(mocks.api.mock.calls.some(([request]) => request.body)).toBe(false)
  })

  it('selecting a preset immediately restores its routes for new chats', async () => {
    state = { names: ['quick', 'deep'], active: 'quick' }
    mount()
    const select = await screen.findByRole('combobox', { name: 'Saved presets' })
    fireEvent.keyDown(select, { key: 'ArrowDown' })
    fireEvent.click(await screen.findByRole('option', { name: 'deep' }))
    await waitFor(() =>
      expect(mocks.api).toHaveBeenCalledWith(expect.objectContaining({ body: { action: 'use', name: 'deep' } }))
    )
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it('sets the next-chat default without dispatching to an existing session', async () => {
    mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Use for new chats' }))
    await waitFor(() =>
      expect(mocks.api).toHaveBeenCalledWith(expect.objectContaining({ body: { action: 'use', name: 'quick' } }))
    )
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it.each([false, true])('suppresses stale apply completion after focus changes (error=%s)', async fail => {
    let finish!: (value: { output?: string; error?: string }) => void
    mocks.request.mockReturnValue(
      new Promise(resolve => {
        finish = resolve
      })
    )
    mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Apply to current chat' }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalled())
    await act(async () => {
      session.set('another-chat')
      finish(fail ? { error: 'old-chat-error' } : { output: 'Applied to original chat' })
    })
    expect(screen.queryByText('Applied to original chat')).toBeNull()
    expect(mocks.error).not.toHaveBeenCalled()
  })

  it('does not carry an earlier apply status into another focused chat', async () => {
    mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Apply to current chat' }))
    await screen.findByText('Applied quick')
    act(() => session.set('another-chat'))
    expect(screen.queryByText('Applied quick')).toBeNull()
  })

  it('applies only after confirmation via slash.exec to the focused current-profile session', async () => {
    mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Apply to current chat' }))
    await screen.findByText('Applied quick')
    expect(mocks.confirm).toHaveBeenCalledWith(
      expect.objectContaining({ description: expect.stringContaining('breaks prompt caching') })
    )
    expect(mocks.request).toHaveBeenCalledWith('slash.exec', {
      session_id: 'runtime-work',
      command: 'delegation preset quick --now'
    })
  })

  it.each(['other-profile', 'other-connection', 'draft'])('refuses apply for %s', async kind => {
    if (kind === 'other-profile') {
      owner.set({ connectionId: 'local', profile: 'other' })
    }

    if (kind === 'other-connection') {
      owner.set({ connectionId: 'remote', profile: 'work' })
    }

    if (kind === 'draft') {
      session.set(null)
    }

    mount()
    const button = (await screen.findByRole('button', { name: 'Apply to current chat' })) as HTMLButtonElement
    expect(button.disabled).toBe(true)
    fireEvent.click(button)
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it('does not retarget when the focused session changes during confirmation', async () => {
    mocks.confirm.mockImplementation(async () => {
      session.set('different')

      return true
    })
    mount()
    fireEvent.click(await screen.findByRole('button', { name: 'Apply to current chat' }))
    await waitFor(() => expect(mocks.confirm).toHaveBeenCalled())
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it('degrades gracefully for an old or disabled plugin while leaving route editing intact', async () => {
    mocks.api.mockRejectedValue(new Error('404 Not Found'))
    mount()
    await screen.findByText('Presets require an enabled, up-to-date Advanced Delegation plugin.')
    expect(screen.queryByRole('button', { name: 'Save current as…' })).toBeNull()
  })

  it('surfaces an authoritative write failure without reporting success', async () => {
    mount()
    await screen.findByRole('button', { name: 'Use for new chats' })
    mocks.api.mockRejectedValueOnce(new Error('409 Managed install'))
    fireEvent.click(screen.getByRole('button', { name: 'Use for new chats' }))
    await waitFor(() => expect(mocks.error).toHaveBeenCalled())
    expect(mocks.request).not.toHaveBeenCalled()
  })
})
