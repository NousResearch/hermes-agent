import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setWorkflowCopilotOpen } from '@/store/workflow'

import { TitlebarControls } from './titlebar-controls'

afterEach(() => {
  cleanup()
  setWorkflowCopilotOpen(false)
  Reflect.deleteProperty(window, 'hermesDesktop')
})

function setDesktopBridge(value: unknown) {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value
  })
}

describe('TitlebarControls workflow mode', () => {
  it('shows a text login entry in the top-right titlebar when no EasyHermes account is signed in', async () => {
    setDesktopBridge({
      account: {
        status: vi.fn(async () => ({ loggedIn: false, cloudBaseUrl: 'https://flow.karivibe.com' }))
      }
    })

    render(
      <MemoryRouter initialEntries={['/sessions']}>
        <TitlebarControls onOpenSettings={() => {}} />
      </MemoryRouter>
    )

    expect(await screen.findByRole('button', { name: '登录 / 注册' })).toBeTruthy()
  })

  it('shows the EasyHermes username after account login', async () => {
    setDesktopBridge({
      account: {
        status: vi.fn(async () => ({
          cloudBaseUrl: 'https://flow.karivibe.com',
          email: 'kari@example.com',
          loggedIn: true,
          username: 'Kari'
        }))
      }
    })

    render(
      <MemoryRouter initialEntries={['/sessions']}>
        <TitlebarControls onOpenSettings={() => {}} />
      </MemoryRouter>
    )

    expect(await screen.findByRole('button', { name: 'Kari' })).toBeTruthy()
  })

  it('reuses the top-right sidebar button as the workflow copilot toggle', () => {
    render(
      <MemoryRouter initialEntries={['/workflow']}>
        <TitlebarControls onOpenSettings={() => {}} />
      </MemoryRouter>
    )

    const appControls = screen.getByLabelText('App controls')
    const buttons = appControls.querySelectorAll('button')
    const rightmostButton = buttons[buttons.length - 1]

    expect(rightmostButton?.getAttribute('aria-label')).toBe('弹出爱马仕 Copilot')
    expect(rightmostButton?.getAttribute('aria-pressed')).toBe('false')

    fireEvent.click(rightmostButton)

    expect(rightmostButton?.getAttribute('aria-label')).toBe('收回爱马仕 Copilot')
    expect(rightmostButton?.getAttribute('aria-pressed')).toBe('true')
  })
})
