// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LocalDashboardAuth } from './local-dashboard-auth'

const mocks = vi.hoisted(() => ({
  configured: '',
  request: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  activeGatewayProfileKey: () => 'default',
  requestGatewayForProfile: (...args: unknown[]) => mocks.request(...args)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        gateway: {
          localDashboardAuthTitle: 'Local dashboard sign-in',
          localDashboardAuthDesc: 'Set the sign-in for the local backend only.',
          localDashboardAuthUsername: 'Dashboard username',
          localDashboardAuthPassword: 'New password',
          localDashboardAuthConfirm: 'Confirm new password',
          localDashboardAuthShowPassword: 'Show password',
          localDashboardAuthHidePassword: 'Hide password',
          localDashboardAuthSave: 'Save dashboard sign-in',
          localDashboardAuthResetForm: 'Reset form',
          localDashboardAuthConfigured: (username: string) => `Sign-in is set for “${username}”.`,
          localDashboardAuthNotConfigured: 'No dashboard sign-in is set yet.',
          localDashboardAuthSaved: 'Dashboard sign-in saved.',
          localDashboardAuthRestartNote: 'Restart the gateway for it to take effect.',
          localDashboardAuthTrustedNote: 'Stored as a hash, trusted networks only.',
          localDashboardAuthLoadFailed: 'Could not load the local dashboard sign-in',
          localDashboardAuthSaveFailed: 'Could not save the dashboard sign-in',
          localDashboardAuthUsernameRequired: 'Enter a dashboard username.',
          localDashboardAuthPasswordTooShort: 'The password must be at least 8 characters.',
          localDashboardAuthPasswordMismatch: 'The passwords do not match.'
        }
      }
    }
  })
}))

describe('LocalDashboardAuth', () => {
  beforeEach(() => {
    mocks.configured = ''
    mocks.request.mockImplementation(async (_profile: unknown, method: string, params: Record<string, unknown>) => {
      if (method === 'config.get' && params.key === 'dashboard.auth') {
        return { value: mocks.configured }
      }

      if (method === 'config.set' && params.key === 'dashboard.auth') {
        return { key: 'dashboard.auth', value: params.username, auth_configured: true, restart_required: true }
      }

      throw new Error(`unexpected call ${method}`)
    })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows the not-configured state and the configured username from the read', async () => {
    render(<LocalDashboardAuth />)
    await waitFor(() => expect(screen.getByText('No dashboard sign-in is set yet.')).toBeTruthy())
    expect(mocks.request).toHaveBeenCalledWith('default', 'config.get', { key: 'dashboard.auth' })
  })

  it('shows a configured username without ever echoing a password or hash', async () => {
    mocks.configured = 'ben'
    render(<LocalDashboardAuth />)
    await waitFor(() => expect(screen.getByText('Sign-in is set for “ben”.')).toBeTruthy())
    expect(document.body.textContent).not.toContain('scrypt$')
  })

  it('refuses mismatched passwords without calling the gateway', async () => {
    render(<LocalDashboardAuth />)
    await waitFor(() => expect(screen.getByText('No dashboard sign-in is set yet.')).toBeTruthy())

    fireEvent.change(screen.getByLabelText('Dashboard username'), { target: { value: 'ben' } })
    fireEvent.change(screen.getByLabelText('New password'), { target: { value: 'correct horse' } })
    fireEvent.change(screen.getByLabelText('Confirm new password'), { target: { value: 'different horse' } })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Save dashboard sign-in' }))
    })

    expect(screen.getByText('The passwords do not match.')).toBeTruthy()
    expect(mocks.request).not.toHaveBeenCalledWith(
      'default',
      'config.set',
      expect.objectContaining({ key: 'dashboard.auth' })
    )
  })

  it('saves username + password and never computes a hash client-side', async () => {
    render(<LocalDashboardAuth />)
    await waitFor(() => expect(screen.getByText('No dashboard sign-in is set yet.')).toBeTruthy())

    fireEvent.change(screen.getByLabelText('Dashboard username'), { target: { value: 'ben' } })
    fireEvent.change(screen.getByLabelText('New password'), { target: { value: 'correct horse' } })
    fireEvent.change(screen.getByLabelText('Confirm new password'), { target: { value: 'correct horse' } })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Save dashboard sign-in' }))
    })

    const setCall = mocks.request.mock.calls.find(call => call[1] === 'config.set')
    expect(setCall).toBeTruthy()
    expect(setCall?.[2]).toEqual({ key: 'dashboard.auth', value: '', username: 'ben', password: 'correct horse' })
    // The payload carries the password for the server to hash; it must not
    // carry a client-computed hash the server would then trust as-is.
    expect(setCall?.[2]).not.toHaveProperty('password_hash')
    await waitFor(() => expect(screen.getByText(/Dashboard sign-in saved\./)).toBeTruthy())
  })
})
