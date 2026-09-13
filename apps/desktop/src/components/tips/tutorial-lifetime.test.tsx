import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { request } = vi.hoisted(() => ({ request: vi.fn(async () => undefined) }))

vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return { $gateway: atom<unknown>(null), activeGateway: () => ({ request }) }
})
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return { $awaitingResponse: atom(false), $busy: atom(false) }
})
vi.mock('react-router', () => ({ useNavigate: () => vi.fn() }))
vi.mock('./local-setup-offer', () => ({ offerLocalSetupTip: () => false }))

import { en } from '@/i18n/en'

const DAY_MS = 24 * 60 * 60_000
const START = new Date('2026-01-01T12:00:00Z').getTime()

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  request.mockClear()
  vi.useFakeTimers()
  vi.setSystemTime(START)
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('retires both tutorial features after 30 days across launches, then keeps a manual re-enable', async () => {
  const { useTipRotation } = await import('./use-tip-rotation')
  let tips = await import('@/store/tips')
  let tours = await import('@/store/tours')
  const firstLaunch = renderHook(() => useTipRotation(en.tips))

  expect(tips.$tipsEnabled.get()).toBe(true)
  expect(tours.$toursEnabled.get()).toBe(true)
  expect(request).not.toHaveBeenCalled()
  firstLaunch.unmount()

  // Reload updated modules against the same persistent storage, just as an
  // app update does. The deadline must remain anchored to the first launch.
  vi.setSystemTime(START + 30 * DAY_MS - 30_000)
  vi.resetModules()
  const returning = await import('./use-tip-rotation')
  tips = await import('@/store/tips')
  tours = await import('@/store/tours')
  const secondLaunch = renderHook(() => returning.useTipRotation(en.tips))

  expect(tips.$tipsEnabled.get()).toBe(true)
  expect(tours.$toursEnabled.get()).toBe(true)
  tips.resetTips() // Replaying the catalog must not restart the month either.
  tips.showTip({ side: 'bottom', targets: ['body'], text: 'A pending tip' })

  act(() => vi.advanceTimersByTime(30_000))

  expect(tips.$tipsEnabled.get()).toBe(false)
  expect(tours.$toursEnabled.get()).toBe(false)
  expect(tips.$activeTip.get()).toBeNull()
  expect(request).toHaveBeenCalledWith('config.set', { key: 'display.in_app_tips', value: 'false' })
  expect(request).toHaveBeenCalledWith('config.set', { key: 'display.in_app_tours', value: 'false' })

  act(() => {
    tips.setTipsEnabled(true)
    tours.setToursEnabled(true)
    tips.resetTips()
  })
  secondLaunch.unmount()

  vi.setSystemTime(START + 60 * DAY_MS)
  vi.resetModules()
  const later = await import('./use-tip-rotation')
  tips = await import('@/store/tips')
  tours = await import('@/store/tours')
  renderHook(() => later.useTipRotation(en.tips))
  act(() => vi.advanceTimersByTime(30_000))

  expect(tips.$tipsEnabled.get()).toBe(true)
  expect(tours.$toursEnabled.get()).toBe(true)
})

it('uses existing tip history on upgrade without treating invalid dates as experience', async () => {
  localStorage.setItem(
    'hermes.desktop.tips.shownAt.v1',
    JSON.stringify({
      old: START - 31 * DAY_MS,
      recent: START - DAY_MS,
      invalid: 'not a timestamp',
      future: START + DAY_MS,
      zero: 0,
      negative: -1
    })
  )
  const { useTipRotation } = await import('./use-tip-rotation')
  let tips = await import('@/store/tips')
  let tours = await import('@/store/tours')
  const upgraded = renderHook(() => useTipRotation(en.tips))

  expect(tips.$tipsEnabled.get()).toBe(false)
  expect(tours.$toursEnabled.get()).toBe(false)
  upgraded.unmount()

  localStorage.clear()
  localStorage.setItem(
    'hermes.desktop.tips.shownAt.v1',
    JSON.stringify({ invalid: 'not a timestamp', future: START + DAY_MS, zero: 0, negative: -1 })
  )
  localStorage.setItem('hermes.desktop.tips.rotation.v1', 'false')
  vi.resetModules()
  const fresh = await import('./use-tip-rotation')
  tips = await import('@/store/tips')
  tours = await import('@/store/tours')
  renderHook(() => fresh.useTipRotation(en.tips))

  // No usable history: begin the month now, never turn an existing Off on.
  expect(tips.$tipsEnabled.get()).toBe(false)
  expect(tours.$toursEnabled.get()).toBe(true)
  act(() => {
    vi.setSystemTime(START + 30 * DAY_MS)
    vi.advanceTimersByTime(30_000)
  })
  expect(tours.$toursEnabled.get()).toBe(false)
})
