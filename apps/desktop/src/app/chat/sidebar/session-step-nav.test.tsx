import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SessionStepNav } from './session-step-nav'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        stepDown: 'Next session',
        stepUp: 'Previous session'
      }
    }
  })
}))

const IDS = ['session-a', 'session-b', 'session-c']

describe('SessionStepNav', () => {
  it('steps to the previous session from a middle row', () => {
    const onStep = vi.fn()
    render(<SessionStepNav activeId="session-b" ids={IDS} onStep={onStep} />)

    fireEvent.click(screen.getByRole('button', { name: 'Previous session' }))

    expect(onStep).toHaveBeenCalledWith('session-a')
  })

  it('steps to the next session from a middle row', () => {
    const onStep = vi.fn()
    render(<SessionStepNav activeId="session-b" ids={IDS} onStep={onStep} />)

    fireEvent.click(screen.getByRole('button', { name: 'Next session' }))

    expect(onStep).toHaveBeenCalledWith('session-c')
  })

  it('disables the up button at the top of the list', () => {
    const onStep = vi.fn()
    render(<SessionStepNav activeId="session-a" ids={IDS} onStep={onStep} />)

    expect(screen.getByRole('button', { name: 'Previous session' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Next session' }).hasAttribute('disabled')).toBe(false)
  })

  it('disables the down button at the bottom of the list', () => {
    const onStep = vi.fn()
    render(<SessionStepNav activeId="session-c" ids={IDS} onStep={onStep} />)

    expect(screen.getByRole('button', { name: 'Previous session' }).hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('button', { name: 'Next session' }).hasAttribute('disabled')).toBe(true)
  })

  it('disables both buttons when no session is active', () => {
    const onStep = vi.fn()
    render(<SessionStepNav activeId={null} ids={IDS} onStep={onStep} />)

    expect(screen.getByRole('button', { name: 'Previous session' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Next session' }).hasAttribute('disabled')).toBe(true)
  })

  it('disables both buttons when the active session is not in the list', () => {
    const onStep = vi.fn()
    render(<SessionStepNav activeId="session-elsewhere" ids={IDS} onStep={onStep} />)

    expect(screen.getByRole('button', { name: 'Previous session' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: 'Next session' }).hasAttribute('disabled')).toBe(true)
  })
})
