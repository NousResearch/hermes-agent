import { render } from '@testing-library/react'
import { beforeEach, expect, it } from 'vitest'

import { $setupHandoff, resetSetupHandoffForTests } from '../setup-profile'

import { HandoffCard } from './build'

beforeEach(resetSetupHandoffForTests)

it('waits for a settled directive and raises the handoff once across remounts', () => {
  const attrs = { task: 'Tracker', brief: 'Build my tracker', plan: 'plugin' }
  const { rerender, unmount } = render(<HandoffCard attrs={attrs} locked />)

  expect($setupHandoff.get()).toBeNull()
  rerender(<HandoffCard attrs={attrs} locked={false} />)
  const requested = $setupHandoff.get()
  expect(requested).toEqual({ ...attrs, phase: 'pending' })
  unmount()
  render(<HandoffCard attrs={attrs} locked={false} />)
  expect($setupHandoff.get()).toBe(requested)
})
