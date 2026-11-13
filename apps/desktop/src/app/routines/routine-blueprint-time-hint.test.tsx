import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { AutomationBlueprint } from '@/hermes'

import { RoutineBlueprintForm } from './routine-blueprint-form'
import { formatClock } from './routine-schedule'
import type { RoutineSource } from './routine-source'

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

afterEach(cleanup)

const blueprint: AutomationBlueprint = {
  appUrl: '',
  category: 'daily',
  command: '/blueprint morning-brief',
  description: 'A short daily briefing.',
  // The backend's own hint for a time slot, which disagreed with the 12-hour field beside it.
  fields: [
    {
      default: '08:00',
      help: '24h local time, e.g. 08:00',
      label: 'What time?',
      name: 'time',
      optional: false,
      options: [],
      type: 'time'
    }
  ],
  key: 'morning-brief',
  tags: [],
  title: 'Morning briefing'
}

const source = { connectionId: null, createScope: (profile: string) => ({ profile }) } as unknown as RoutineSource

it('hints a ready-made routine’s time in the same clock the time field shows', () => {
  render(
    <RoutineBlueprintForm
      blueprint={blueprint}
      lockProfile
      onCancel={vi.fn()}
      onDone={vi.fn()}
      onProfileChange={vi.fn()}
      profile="default"
      source={source}
    />
  )

  expect(screen.queryByText(/24h/)).toBeNull()
  // en-US: the field reads 08:00 AM, so the hint's example is a 12-hour clock too.
  expect(formatClock('08:00', 'en')).toMatch(/^8:00\sAM$/)
  expect(screen.getByText(`Your local time, e.g. ${formatClock('08:00', 'en')}`)).toBeTruthy()
})
