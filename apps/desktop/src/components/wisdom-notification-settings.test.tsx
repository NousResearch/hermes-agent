import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
// @vitest-environment jsdom
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { WisdomNotificationSettings } from './wisdom-notification-settings'

const { prepare, read, choose } = vi.hoisted(() => ({ prepare: vi.fn(), read: vi.fn(), choose: vi.fn() }))
vi.mock('@/hermes', () => ({ prepareWisdomMute: prepare, getWisdomMute: read, chooseWisdomMute: choose }))
vi.mock('@/api/client', () => ({
  capabilityScoped: (profile: string) => ({ profile, connectionId: 'local' }),
  profileScopeKey: (scope: object) => JSON.stringify(scope)
}))

const control = {
  id: 'a'.repeat(32),
  expires_at: Date.now() / 1000 + 600,
  organization_id: 'org',
  sync: null,
  mute: { org_id: 'org', revision: 3, duration: null, muted: false, forever: false, muted_until: null }
}

const snapshot = { organization_id: 'org', gateway_available: true, mute: control.mute, sync: null }
beforeEach(() => {
  vi.clearAllMocks()
  prepare.mockResolvedValue(control)
  read.mockResolvedValue(snapshot)
  choose.mockResolvedValue(snapshot)
})
afterEach(cleanup)

async function open() {
  fireEvent.click(screen.getByRole('button', { name: 'Notification settings' }))
  await screen.findByText('Notifications on', { selector: 'p' })
}

describe('Wisdom notification settings', () => {
  it('loads only on request and changes only after Save', async () => {
    render(<WisdomNotificationSettings profile="demo" />)
    expect(prepare).not.toHaveBeenCalled()
    await open()
    fireEvent.change(screen.getByRole('combobox'), { target: { value: '1_week' } })
    expect(choose).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(choose).toHaveBeenCalledOnce())
    expect(choose).toHaveBeenCalledWith(control.id, '1_week', { profile: 'demo', connectionId: 'local' })
  })

  it('keeps the original choice when a reply is lost', async () => {
    choose.mockRejectedValueOnce(new Error('timeout'))
    render(<WisdomNotificationSettings profile="demo" />)
    await open()
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'forever' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByRole('button', { name: 'Retry' })
    expect((screen.getByRole('combobox') as HTMLSelectElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(choose).toHaveBeenCalledTimes(2))
    expect(choose.mock.calls[0]).toEqual(choose.mock.calls[1])
  })

  it('does not hide shared state on a transient poll error', async () => {
    render(<WisdomNotificationSettings profile="demo" />)
    await open()
    read.mockRejectedValue(new Error('offline'))
    fireEvent(window, new Event('focus'))
    await screen.findByRole('alert')
    expect(screen.getByText('Notifications on', { selector: 'p' })).toBeTruthy()
  })

  it('remains usable under StrictMode and separates pending from shared state', async () => {
    prepare.mockResolvedValue({
      ...control,
      sync: {
        preference_sync: 'pending',
        requested_duration: 'forever',
        mutation_id: 'queued'
      }
    })
    render(
      <StrictMode>
        <WisdomNotificationSettings profile="demo" />
      </StrictMode>
    )
    await open()
    expect(screen.getByText('Your choice is saved locally and waiting to sync.')).toBeTruthy()
    expect(choose).not.toHaveBeenCalled()
    expect(screen.getByText('Notifications on', { selector: 'p' })).toBeTruthy()
  })

  it('drops responses from the previous profile', async () => {
    let finish!: (value: typeof control) => void
    prepare.mockReturnValueOnce(
      new Promise(resolve => {
        finish = resolve
      })
    )
    const { rerender } = render(<WisdomNotificationSettings profile="first" />)
    fireEvent.click(screen.getByRole('button', { name: 'Notification settings' }))
    await waitFor(() => expect(prepare).toHaveBeenCalledOnce())
    rerender(<WisdomNotificationSettings profile="second" />)
    await act(async () => finish(control))
    expect(screen.queryByRole('combobox')).toBeNull()
    await open()
    expect(prepare).toHaveBeenLastCalledWith({ profile: 'second', connectionId: 'local' })
  })
})
