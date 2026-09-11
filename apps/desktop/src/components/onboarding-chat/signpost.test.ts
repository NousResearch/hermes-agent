import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { startTour } from '@/lib/tour'

import { showHandoffTour } from './signpost'

vi.mock('@/i18n', () => ({ translateNow: (key: string) => key }))
vi.mock('@/lib/tour', () => ({ startTour: vi.fn().mockResolvedValue(undefined) }))

beforeEach(() => {
  vi.useFakeTimers()
  document.body.innerHTML = `
    <section id="rail-pane"><div data-tour="profile-rail"></div></section>
    <section id="sessions-pane"><div data-tour="sessions-sidebar"></div></section>
  `
})

afterEach(() => {
  document.body.innerHTML = ''
  vi.restoreAllMocks()
  vi.clearAllMocks()
  vi.useRealTimers()
})

it.each(['width', 'height', 'pane'])('waits until the rail is visible when hidden by %s', async hiddenBy => {
  const rail = document.querySelector('[data-tour="profile-rail"]')!
  const sessions = document.querySelector('[data-tour="sessions-sidebar"]')!
  let railBounds = new DOMRect(0, 0, hiddenBy === 'width' ? 0 : 200, hiddenBy === 'height' ? 0 : 600)
  vi.spyOn(rail, 'getBoundingClientRect').mockImplementation(() => railBounds)
  vi.spyOn(sessions, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 200, 600))

  if (hiddenBy === 'pane') {
    rail.parentElement!.setAttribute('data-pane-hidden', '')
  }

  const tour = showHandoffTour()
  await vi.advanceTimersByTimeAsync(120)
  expect(startTour).not.toHaveBeenCalled()

  railBounds = new DOMRect(0, 0, 200, 600)
  rail.parentElement!.removeAttribute('data-pane-hidden')
  await vi.advanceTimersByTimeAsync(120)
  await tour

  expect(startTour).toHaveBeenCalledTimes(1)
  expect(vi.mocked(startTour).mock.calls[0][0].map(step => step.selector)).toEqual([
    '[data-tour="profile-rail"]',
    '[data-tour="sessions-sidebar"]',
    '[data-tour="profile-rail"]'
  ])
})

it('shows the two rail steps when the sessions pane stays hidden', async () => {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 200, 600))
  document.querySelector('#sessions-pane')!.setAttribute('data-pane-hidden', '')

  const tour = showHandoffTour()
  await vi.advanceTimersByTimeAsync(1600)
  await tour

  expect(startTour).toHaveBeenCalledTimes(1)
  expect(vi.mocked(startTour).mock.calls[0][0].map(step => step.text)).toEqual([
    'handoffTour.profileText',
    'handoffTour.stayText'
  ])
})
