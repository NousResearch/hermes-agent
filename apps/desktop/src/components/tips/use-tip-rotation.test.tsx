import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $awaitingResponse, $busy, $connection } from '@/store/session'
import { $activeTip, $nextTipAt, $retiredTips, $tipsEnabled, $tipShownAt } from '@/store/tips'

import { offerLocalSetupTip } from './local-setup-offer'
import { useTipRotation } from './use-tip-rotation'

vi.mock('./local-setup-offer', () => ({ offerLocalSetupTip: vi.fn(() => false) }))
vi.mock('@/store/tutorial-lifetime', () => ({ checkTutorialLifetime: vi.fn() }))

const reply = async (request: { path: string }) =>
  request.path.endsWith('/jobs')
    ? { jobs: [] }
    : {
        enabled: true,
        runtime_installed: true,
        update_available: true,
        configured_tag: 'target'
      }

const api = vi.fn(reply)

function Harness() {
  useTipRotation(en.tips)

  return <button data-tour="model-pill">Model</button>
}

function mount() {
  return render(
    <MemoryRouter>
      <Harness />
    </MemoryRouter>
  )
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(Math, 'random').mockReturnValue(0)
  vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(true)
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 160, 32))
  $connection.set({ mode: 'local' } as never)
  $localModelsEnabled.set(true)
  $tipsEnabled.set(true)
  $activeTip.set(null)
  $nextTipAt.set(null)
  $retiredTips.set([])
  $tipShownAt.set({})
  $busy.set(false)
  $awaitingResponse.set(false)
  vi.clearAllMocks()
  api.mockReset().mockImplementation(reply)
  window.hermesDesktop = { api } as never
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})
it('offers the update at the first quiet moment despite tutorial settling and cooldown', async () => {
  $nextTipAt.set(Date.now() + 6 * 60 * 60_000)
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_000)
  })
  expect($activeTip.get()?.action?.label).toBe('Update now')
  expect(offerLocalSetupTip).not.toHaveBeenCalled()
  expect(api.mock.calls.some(([request]) => 'method' in request && request.method === 'POST')).toBe(false)
})

it('does not turn the quiet-moment check into constant backend polling', async () => {
  api.mockImplementation(async request =>
    request.path.endsWith('/jobs')
      ? { jobs: [] }
      : {
          enabled: true,
          runtime_installed: true,
          update_available: false,
          configured_tag: 'target'
        }
  )
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(50_000)
  })
  expect(api.mock.calls.filter(([request]) => request.path.endsWith('/status'))).toHaveLength(1)
  expect($activeTip.get()).toBeNull()
})

it('retries a transient startup failure within a few seconds', async () => {
  api.mockRejectedValueOnce(new Error('backend starting'))
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(8_000)
  })
  expect($activeTip.get()?.action?.label).toBe('Update now')
})

it('waits for an active response and typing to stop', async () => {
  $busy.set(true)
  $awaitingResponse.set(true)
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3_000)
  })
  expect(api).not.toHaveBeenCalled()
  $busy.set(false)
  $awaitingResponse.set(false)
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(4_000)
  })
  expect($activeTip.get()).toBeNull()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_000)
  })
  expect($activeTip.get()?.action?.label).toBe('Update now')
})

it('waits for a focused chat with no menu open', async () => {
  const focus = vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(false)
  const menu = globalThis.document.createElement('div')
  menu.setAttribute('role', 'menu')
  globalThis.document.body.append(menu)
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3_000)
  })
  focus.mockReturnValue(true)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3_000)
  })
  expect(api).not.toHaveBeenCalled()
  menu.remove()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_000)
  })
  expect($activeTip.get()?.action?.label).toBe('Update now')
})

it('does not consume the update before the model pill is visible', async () => {
  const geometry = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect())
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3_000)
  })
  expect(api).not.toHaveBeenCalled()
  expect($tipShownAt.get()).toEqual({})
  geometry.mockReturnValue(new DOMRect(0, 0, 160, 32))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_000)
  })
  expect($activeTip.get()?.action?.label).toBe('Update now')
})

it.each(['flag-off', 'tips-off', 'remote'])('does not read or offer when %s', async guard => {
  if (guard === 'flag-off') {
    $localModelsEnabled.set(false)
  }

  if (guard === 'tips-off') {
    $tipsEnabled.set(false)
  }

  if (guard === 'remote') {
    $connection.set({ mode: 'remote' } as never)
  }
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3_000)
  })
  expect(api).not.toHaveBeenCalled()
  expect($activeTip.get()).toBeNull()
})

it('keeps tutorial settling and cooldown when no engine update is available', async () => {
  api.mockImplementation(async request =>
    request.path.endsWith('/jobs')
      ? { jobs: [] }
      : {
          enabled: true,
          runtime_installed: true,
          update_available: false,
          configured_tag: 'target'
        }
  )
  mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(4 * 60_000)
  })
  expect(offerLocalSetupTip).not.toHaveBeenCalled()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(60_000)
  })
  expect(offerLocalSetupTip).toHaveBeenCalledTimes(1)
  $nextTipAt.set(Date.now() + 6 * 60 * 60_000)
  $activeTip.set(null)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(60_000)
  })
  expect(offerLocalSetupTip).toHaveBeenCalledTimes(1)
})

it('stops the fast check when the host unmounts', async () => {
  api.mockImplementation(async request =>
    request.path.endsWith('/jobs')
      ? { jobs: [] }
      : {
          enabled: true,
          runtime_installed: true,
          update_available: false,
          configured_tag: 'target'
        }
  )
  const view = mount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_000)
  })
  const calls = api.mock.calls.length
  view.unmount()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(120_000)
  })
  expect(api).toHaveBeenCalledTimes(calls)
  expect(vi.getTimerCount()).toBe(0)
})
