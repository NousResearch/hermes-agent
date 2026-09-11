import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, expect, it } from 'vitest'

import { $handoffError } from '@/app/contrib/handoff-receipt'

import { $setupHandoff, resetSetupHandoffForTests } from '../setup-profile'

import { HandoffCard } from './build'

beforeEach(() => {
  resetSetupHandoffForTests()
  $handoffError.set(null)
})

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

it('shows the actual failure and a deliberate retry rather than claiming an alternate build', () => {
  $setupHandoff.set({ task: 'Tracker', brief: 'Build my tracker', plan: 'build', phase: 'error' })
  $handoffError.set('Provider unavailable')
  render(<HandoffCard attrs={{ task: 'Tracker', brief: 'Build my tracker' }} locked={false} />)
  expect(screen.queryByText(/building here instead/)).toBeNull()
  expect(screen.getByText('Provider unavailable')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Retry first build' }))
  expect($setupHandoff.get()?.phase).toBe('pending')
  expect(screen.queryByRole('button', { name: 'Retry first build' })).toBeNull()
})
