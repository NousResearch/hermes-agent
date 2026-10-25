import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib'
import { CHAT_WELCOME_AREA } from '@/lib/chat-welcome'

import { ChatWelcomeSlot } from './chat-welcome-slot'

const disposers: (() => void)[] = []

afterEach(() => {
  for (const dispose of disposers.splice(0)) {
    dispose()
  }
})

describe('draft chat welcome slot', () => {
  it('renders the first valid contribution with the draft cwd and profile', () => {
    let payload: unknown
    const laterRender = vi.fn(() => <div>later welcome</div>)
    disposers.push(
      registry.register({ area: CHAT_WELCOME_AREA, data: {}, id: 'invalid' }),
      registry.register({
        area: CHAT_WELCOME_AREA,
        data: {
          render: (props: unknown) => {
            payload = props

            return <div data-testid="welcome">plugin welcome</div>
          }
        },
        id: 'welcome'
      }),
      registry.register({ area: CHAT_WELCOME_AREA, data: { render: laterRender }, id: 'later-welcome' })
    )

    render(
      <ChatWelcomeSlot cwd="/work/research" fallback={<div data-testid="intro">core intro</div>} profile="lab" />
    )

    expect(payload).toEqual({ cwd: '/work/research', profile: 'lab' })
    expect(screen.getByTestId('welcome').textContent).toBe('plugin welcome')
    expect(screen.queryByTestId('intro')).toBeNull()
    expect(laterRender).not.toHaveBeenCalled()
  })

  it('renders the core Intro fallback when no valid contribution is registered', () => {
    disposers.push(registry.register({ area: CHAT_WELCOME_AREA, data: {}, id: 'invalid' }))

    render(<ChatWelcomeSlot cwd="/work/research" fallback={<div data-testid="intro">core intro</div>} profile="lab" />)

    expect(screen.getByTestId('intro').textContent).toBe('core intro')
  })

  it('fails open to the core Intro when the claimed contribution throws', () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    disposers.push(
      registry.register({
        area: CHAT_WELCOME_AREA,
        data: {
          render: () => {
            throw new Error('welcome exploded')
          }
        },
        id: 'broken-welcome'
      })
    )

    render(<ChatWelcomeSlot cwd="/work/research" fallback={<div data-testid="intro">core intro</div>} profile="lab" />)

    expect(screen.getByTestId('intro').textContent).toBe('core intro')
    expect(error).toHaveBeenCalled()
    error.mockRestore()
  })
})
