import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $remoteOverrideDialogProfile, openRemoteOverrideDialog } from '@/store/profile-remote-override'

import { ProfileRemoteOverrideDialog } from './profile-remote-override-dialog'

// The rail affordance that writes connection.json `profiles.<name>` through
// the typed Electron bridge (#91349): the renderer must never touch the file
// itself, the token must ride the safeStorage path (applyConnectionConfig),
// a first-time connect shows the risk confirmation, keyring-less machines
// require the plain-text opt-in, and a registry name collision warns.

const getConnectionConfig = vi.fn()
const applyConnectionConfig = vi.fn()
const list = vi.fn()

const localScope = {
  cloudOrg: '',
  envOverride: false,
  mode: 'local',
  profile: 'work',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPlainText: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: '',
  secureTokenStorage: true
}

const emptyRegistry = { connections: [], primary: 'local', secureTokenStorage: true, version: 2 }

beforeEach(() => {
  getConnectionConfig.mockResolvedValue(localScope)
  applyConnectionConfig.mockResolvedValue({ ...localScope, mode: 'remote', remoteUrl: 'https://box.example.com' })
  list.mockResolvedValue(emptyRegistry)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { applyConnectionConfig, connections: { list }, getConnectionConfig }
  })
  $remoteOverrideDialogProfile.set(null)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function openDialog(profile = 'work') {
  render(<ProfileRemoteOverrideDialog profileNames={[profile]} />)
  openRemoteOverrideDialog(profile)
  await waitFor(() => expect(getConnectionConfig).toHaveBeenCalledWith(profile))
  await screen.findByLabelText(/Remote address/)
}

describe('ProfileRemoteOverrideDialog', () => {
  it('walks the first-time connect through the risk confirmation before writing', async () => {
    await openDialog()

    fireEvent.change(screen.getByLabelText(/Remote address/), { target: { value: 'https://box.example.com' } })
    fireEvent.change(screen.getByLabelText(/Access token/), { target: { value: 'tok-123' } })
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    // No write yet — the risk note comes first.
    expect(applyConnectionConfig).not.toHaveBeenCalled()
    expect(screen.getByText('Connect this profile to a remote host?')).toBeTruthy()
    expect(screen.getByText(/Only connect to a host you trust/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await waitFor(() =>
      expect(applyConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        profile: 'work',
        remoteAuthMode: 'token',
        remoteToken: 'tok-123',
        remoteUrl: 'https://box.example.com'
      })
    )
    await waitFor(() => expect($remoteOverrideDialogProfile.get()).toBeNull())
  })

  it('skips the confirmation when editing an existing override (token rotation)', async () => {
    getConnectionConfig.mockResolvedValue({
      ...localScope,
      mode: 'remote',
      remoteTokenSet: true,
      remoteUrl: 'https://box.example.com'
    })
    await openDialog()

    fireEvent.change(screen.getByLabelText(/Access token/), { target: { value: 'rotated-tok' } })
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await waitFor(() =>
      expect(applyConnectionConfig).toHaveBeenCalledWith(
        expect.objectContaining({ profile: 'work', remoteToken: 'rotated-tok' })
      )
    )
  })

  it('offers removal for an overridden profile and clears via mode local', async () => {
    getConnectionConfig.mockResolvedValue({
      ...localScope,
      mode: 'remote',
      remoteTokenSet: true,
      remoteUrl: 'https://box.example.com'
    })
    applyConnectionConfig.mockResolvedValue(localScope)
    await openDialog()

    fireEvent.click(screen.getByRole('button', { name: 'Remove remote connection' }))

    await waitFor(() => expect(applyConnectionConfig).toHaveBeenCalledWith({ mode: 'local', profile: 'work' }))
    await waitFor(() => expect($remoteOverrideDialogProfile.get()).toBeNull())
  })

  it('requires the explicit unencrypted-token opt-in on keyring-less machines', async () => {
    getConnectionConfig.mockResolvedValue({ ...localScope, secureTokenStorage: false })
    await openDialog()

    fireEvent.change(screen.getByLabelText(/Remote address/), { target: { value: 'https://box.example.com' } })
    fireEvent.change(screen.getByLabelText(/Access token/), { target: { value: 'tok-123' } })

    const connect = screen.getByRole('button', { name: 'Connect' })
    expect((connect as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText(/saved unencrypted on disk/)).toBeTruthy()

    fireEvent.click(screen.getByRole('checkbox'))
    expect((screen.getByRole('button', { name: 'Connect' }) as HTMLButtonElement).disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await waitFor(() =>
      expect(applyConnectionConfig).toHaveBeenCalledWith(expect.objectContaining({ allowPlainTextToken: true }))
    )
  })

  it('warns when the profile name collides with a registered gateway', async () => {
    list.mockResolvedValue({
      ...emptyRegistry,
      connections: [{ id: 'abc123', kind: 'remote', label: 'Work', tokenPreview: null, tokenSet: true }]
    })
    await openDialog('work')

    await screen.findByText(/A gateway named “Work” already exists in Settings/)
  })

  it('keeps the dialog open and shows the error when the write fails', async () => {
    getConnectionConfig.mockResolvedValue({
      ...localScope,
      mode: 'remote',
      remoteTokenSet: true,
      remoteUrl: 'https://box.example.com'
    })
    applyConnectionConfig.mockRejectedValue(new Error('Remote gateway session token is required.'))
    await openDialog()

    fireEvent.change(screen.getByLabelText(/Access token/), { target: { value: 'bad' } })
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await screen.findByText('Remote gateway session token is required.')
    expect($remoteOverrideDialogProfile.get()).toBe('work')
  })
})
