import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $hapticsMuted } from '@/store/haptics'
import { $fileBrowserOpen, $panesFlipped, $sidebarOpen } from '@/store/layout'
import { $activeGatewayProfile } from '@/store/profile'

import { TitlebarControls } from './titlebar-controls'

const navigateMock = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<any>('react-router-dom')

  return {
    ...actual,
    useNavigate: () => navigateMock
  }
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        gateway: {
          title: 'Gateway Connection',
          localTitle: 'Local gateway',
          remoteTitle: 'Remote gateway',
          cloudTitle: 'Hermes Cloud',
          incompleteTitle: 'Remote gateway incomplete',
          incompleteSignIn: 'Enter a remote URL and sign in before switching to remote.',
          incompleteToken: 'Enter a remote URL and session token before switching to remote.',
          cloudNeedsSignIn: 'Sign in to Hermes Cloud to discover the agents on your account.',
          restartingTitle: 'Gateway connection restarting',
          restartingMessage: 'Reconnecting...',
          applyFailed: 'Apply failed',
          signInFailed: 'Sign-in failed'
        }
      },
      boot: {
        failure: {
          signInIncompleteTitle: 'Sign-in incomplete',
          signInIncompleteMessage: 'Please sign in.'
        }
      },
      shell: {
        windowControls: 'Window Controls',
        paneControls: 'Pane Controls',
        appControls: 'App Controls'
      },
      titlebar: {
        hideSidebar: 'Hide sidebar',
        showSidebar: 'Show sidebar',
        swapSidebarSides: 'Swap sides',
        swapSidebarSidesTitle: 'Swap sidebar sides',
        hideRightSidebar: 'Hide right sidebar',
        showRightSidebar: 'Show right sidebar',
        unmuteHaptics: 'Unmute haptics',
        muteHaptics: 'Mute haptics',
        openKeybinds: 'Keybinds',
        openSettings: 'Settings'
      }
    }
  })
}))

const notifyMock = vi.fn()
const notifyErrorMock = vi.fn()
vi.mock('@/store/notifications', () => ({
  notify: (...args: any[]) => notifyMock(...args),
  notifyError: (...args: any[]) => notifyErrorMock(...args)
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

const getConnectionConfig = vi.fn()
const applyConnectionConfig = vi.fn()
const oauthLoginConnectionConfig = vi.fn()
const onConnectionApplied = vi.fn(() => vi.fn())

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()

  window.hermesDesktop = {
    getConnectionConfig,
    applyConnectionConfig,
    oauthLoginConnectionConfig,
    onConnectionApplied
  } as any
})

beforeEach(() => {
  $hapticsMuted.set(false)
  $fileBrowserOpen.set(false)
  $sidebarOpen.set(true)
  $panesFlipped.set(false)
  $activeGatewayProfile.set('default')

  getConnectionConfig.mockResolvedValue({
    mode: 'local',
    remoteUrl: '',
    remoteAuthMode: 'token',
    remoteTokenSet: false,
    remoteTokenPreview: null,
    envOverride: false,
    profile: null,
    remoteOauthConnected: false,
    cloudOrg: ''
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('TitlebarControls with GatewayToggle', () => {
  it('renders correctly and loads active gateway mode config', async () => {
    render(
      <MemoryRouter>
        <TitlebarControls onOpenSettings={vi.fn()} />
      </MemoryRouter>
    )

    await waitFor(() => expect(getConnectionConfig).toHaveBeenCalledWith('default'))
    const button = await screen.findByRole('button', { name: 'Gateway Connection' })
    expect(button).toBeDefined()
  })

  it('can open dropdown menu and select a mode', async () => {
    getConnectionConfig.mockResolvedValueOnce({
      mode: 'local',
      remoteUrl: 'http://localhost:8000',
      remoteAuthMode: 'token',
      remoteTokenSet: true,
      remoteTokenPreview: 'abc...',
      envOverride: false,
      profile: null,
      remoteOauthConnected: false,
      cloudOrg: ''
    })

    render(
      <MemoryRouter>
        <TitlebarControls onOpenSettings={vi.fn()} />
      </MemoryRouter>
    )

    const button = await screen.findByRole('button', { name: 'Gateway Connection' })
    fireEvent.pointerDown(button, { button: 0, ctrlKey: false, pointerType: 'mouse' })

    const localItem = await screen.findByRole('menuitem', { name: 'Local gateway' })
    const remoteItem = await screen.findByRole('menuitem', { name: 'Remote gateway' })
    const cloudItem = await screen.findByRole('menuitem', { name: 'Hermes Cloud' })

    expect(localItem).toBeDefined()
    expect(remoteItem).toBeDefined()
    expect(cloudItem).toBeDefined()

    // Trigger switch to remote
    applyConnectionConfig.mockResolvedValueOnce({
      mode: 'remote',
      remoteUrl: 'http://localhost:8000',
      remoteAuthMode: 'token',
      remoteTokenSet: true,
      remoteTokenPreview: 'abc...',
      envOverride: false,
      profile: null,
      remoteOauthConnected: false,
      cloudOrg: ''
    })

    fireEvent.click(remoteItem)

    await waitFor(() => expect(applyConnectionConfig).toHaveBeenCalledWith({
      mode: 'remote',
      profile: undefined,
      remoteAuthMode: 'token',
      remoteUrl: 'http://localhost:8000',
      cloudOrg: ''
    }))

    expect(notifyMock).toHaveBeenCalledWith(expect.objectContaining({
      title: 'Gateway connection restarting'
    }))
  })

  it('deep-links to settings if remote mode is selected but not configured', async () => {
    // remoteUrl is empty, token not set
    getConnectionConfig.mockResolvedValueOnce({
      mode: 'local',
      remoteUrl: '',
      remoteAuthMode: 'token',
      remoteTokenSet: false,
      remoteTokenPreview: null,
      envOverride: false,
      profile: null,
      remoteOauthConnected: false,
      cloudOrg: ''
    })

    render(
      <MemoryRouter>
        <TitlebarControls onOpenSettings={vi.fn()} />
      </MemoryRouter>
    )

    const button = await screen.findByRole('button', { name: 'Gateway Connection' })
    fireEvent.pointerDown(button, { button: 0, ctrlKey: false, pointerType: 'mouse' })

    const remoteItem = await screen.findByRole('menuitem', { name: 'Remote gateway' })
    fireEvent.click(remoteItem)

    // applyConnectionConfig should NOT have been called
    expect(applyConnectionConfig).not.toHaveBeenCalled()
    // It should have routed to the settings page
    expect(navigateMock).toHaveBeenCalledWith('/settings?tab=gateway')
    expect(notifyMock).toHaveBeenCalledWith(expect.objectContaining({
      kind: 'warning',
      title: 'Remote gateway incomplete'
    }))
  })

  it('triggers OAuth login window and retries apply when apply fails with needsOauthLogin', async () => {
    getConnectionConfig.mockResolvedValue({
      mode: 'local',
      remoteUrl: 'http://my-oauth-gateway.local',
      remoteAuthMode: 'oauth',
      remoteTokenSet: false,
      remoteTokenPreview: null,
      envOverride: false,
      profile: null,
      remoteOauthConnected: true,
      cloudOrg: ''
    })

    const oauthError = new Error('Re-auth required') as any
    oauthError.needsOauthLogin = true
    applyConnectionConfig.mockRejectedValueOnce(oauthError)
    oauthLoginConnectionConfig.mockResolvedValueOnce({ connected: true })
    applyConnectionConfig.mockResolvedValueOnce({
      mode: 'remote',
      remoteUrl: 'http://my-oauth-gateway.local',
      remoteAuthMode: 'oauth',
      remoteTokenSet: false,
      remoteTokenPreview: null,
      envOverride: false,
      profile: null,
      remoteOauthConnected: true,
      cloudOrg: ''
    })

    render(
      <MemoryRouter>
        <TitlebarControls onOpenSettings={vi.fn()} />
      </MemoryRouter>
    )

    const button = await screen.findByRole('button', { name: 'Gateway Connection' })
    fireEvent.pointerDown(button, { button: 0, ctrlKey: false, pointerType: 'mouse' })

    const remoteItem = await screen.findByRole('menuitem', { name: 'Remote gateway' })
    fireEvent.click(remoteItem)

    await waitFor(() => expect(applyConnectionConfig).toHaveBeenCalled())
    await waitFor(() => expect(oauthLoginConnectionConfig).toHaveBeenCalledWith('http://my-oauth-gateway.local'))
    await waitFor(() => expect(applyConnectionConfig).toHaveBeenCalledTimes(2))

    expect(notifyMock).toHaveBeenCalledWith(expect.objectContaining({
      title: 'Gateway connection restarting'
    }))
  })
})
