import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { type RoutinesCopy, RoutinesPanel } from './routines-panel'

const copy: RoutinesCopy = {
  title: 'Routines',
  locked: 'Locked',
  schedule: 'Schedule',
  timezone: 'Timezone',
  destination: 'Destination',
  activate: 'Activate',
  pause: 'Pause',
  saving: 'Saving'
}

const routine = {
  id: 'digest',
  name: 'Digest',
  prompt: 'Write a digest.',
  schedule: '0 9 * * 1',
  state: 'paused' as const,
  timezone: 'Etc/UTC',
  destination: null
}

describe('RoutinesPanel form state', () => {
  it('preserves dirty fields when a status refresh returns the same routine', () => {
    const props = {
      canActivate: true,
      copy,
      onActivate: vi.fn(async () => undefined),
      onPause: vi.fn(async () => undefined)
    }

    const view = render(<RoutinesPanel {...props} routines={[routine]} />)
    const destination = screen.getByLabelText('Digest Destination') as HTMLInputElement

    fireEvent.change(destination, { target: { value: 'discord:research' } })
    view.rerender(
      <RoutinesPanel
        {...props}
        routines={[{ ...routine, schedule: '0 10 * * 1', destination: 'server:stale' }]}
      />
    )

    expect(destination.value).toBe('discord:research')
    expect((screen.getByLabelText('Digest Schedule') as HTMLInputElement).value).toBe('0 9 * * 1')
  })
})
