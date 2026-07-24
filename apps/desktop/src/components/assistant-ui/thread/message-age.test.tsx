import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MessageAge } from './message-age'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      agents: {
        ageDays: (days: number) => `${days}d ago`,
        ageHours: (hours: number) => `${hours}h ago`,
        ageMinutes: (minutes: number) => `${minutes}m ago`,
        ageNow: 'now',
        ageSeconds: (seconds: number) => `${seconds}s ago`
      },
      assistant: {
        thread: {
          today: (time: string) => `Today at ${time}`,
          yesterday: (time: string) => `Yesterday at ${time}`
        }
      }
    }
  })
}))

const tipTrigger = (el: HTMLElement) => el.closest('[data-slot="tooltip-trigger"]')

describe('MessageAge', () => {
  it('renders a focusable Tip trigger with relative and exact time', async () => {
    const createdAt = new Date()
    createdAt.setMinutes(createdAt.getMinutes() - 5)

    render(<MessageAge createdAt={createdAt} />)

    const age = screen.getByText('5m ago')
    expect(age.tagName).toBe('TIME')
    expect(age.getAttribute('datetime')).toBe(createdAt.toISOString())
    expect(age.getAttribute('aria-label')).toMatch(/^5m ago, Today at /)
    expect(age.getAttribute('tabindex')).toBe('0')
    expect(age.getAttribute('title')).toBeNull()
    expect(tipTrigger(age)).toBeTruthy()

    fireEvent.pointerMove(age, { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toMatch(/^Today at /)
  })

  it('does not render an invalid timestamp', () => {
    const { container } = render(<MessageAge createdAt="not-a-date" />)

    expect(container.innerHTML).toBe('')
  })
})
