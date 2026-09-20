import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { SETTINGS_ROUTE } from '@/app/routes'
import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import { $activeGatewayRoute } from '@/store/gateway'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { $connection } from '@/store/session'
import { $activeTip, $nextTipAt, $retiredTips, $tipsEnabled, $tipShownAt, retireActiveTip } from '@/store/tips'

import { offerLocalRuntimeUpdateTip } from './local-runtime-update-offer'
import { TipBubble } from './tip-bubble'

const api = vi.hoisted(() => vi.fn())

const eligible = { enabled: true, runtime_installed: true, update_available: true, configured_tag: 'next-build' }
const anchor = document.createElement('button')
let offer: () => boolean

function Harness() {
  const tip = useStore($activeTip)
  const navigate = useNavigate()
  const location = useLocation()
  offer = () => offerLocalRuntimeUpdateTip(en.tips, () => navigate(`${SETTINGS_ROUTE}?tab=providers&pview=local`))

  return (
    <>
      <output>
        {location.pathname}
        {location.search}
      </output>
      {tip && <TipBubble {...tip} anchor={anchor} onClose={retireActiveTip} />}
    </>
  )
}

async function due() {
  await act(async () => {
    offer()
  })
  await act(async () => {
    offer()
  })
}

beforeEach(async () => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
  vi.clearAllMocks()
  $localModelsEnabled.set(true)
  $connection.set({ mode: 'local' } as never)
  $localRuntimeJobs.set([])
  $activeTip.set(null)
  $retiredTips.set([])
  $tipShownAt.set({})
  $nextTipAt.set(null)
  $tipsEnabled.set(true)
  window.hermesDesktop = { api } as never
  api.mockImplementation(async request => {
    if (request.path.endsWith('/status')) {
      return eligible
    }

    if (request.path.endsWith('/jobs')) {
      return { jobs: [] }
    }

    return { job_id: 'install', backend: 'cpu', tag: 'next-build' }
  })
  await act(async () => {
    render(
      <MemoryRouter>
        <I18nProvider>
          <Harness />
        </I18nProvider>
      </MemoryRouter>
    )
  })
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})
it('discards a completed read that waited too long for a quiet tick', async () => {
  await act(async () => {
    offer()
  })
  const now = Date.now()
  const clock = vi.spyOn(Date, 'now').mockReturnValue(now + 120_000)
  api.mockImplementation(async request =>
    request.path.endsWith('/jobs') ? { jobs: [] } : { ...eligible, update_available: false }
  )
  await act(async () => {
    offer()
  })
  expect($activeTip.get()).toBeNull()
  await act(async () => {
    offer()
  })
  expect($activeTip.get()).toBeNull()
  clock.mockRestore()
})
it('lets rotation proceed after a failed eligibility read and retries at a later due moment', async () => {
  api.mockRejectedValue(new Error('offline'))
  await act(async () => {
    offer()
  })
  let held = true
  await act(async () => {
    held = offer()
  })
  expect(held).toBe(false)
  expect($activeTip.get()).toBeNull()
  api.mockImplementation(async request => (request.path.endsWith('/jobs') ? { jobs: [] } : eligible))
  vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 6_000)
  await due()
  expect($activeTip.get()?.action?.label).toBe('Update now')
})
it('rechecks backend jobs at the due moment instead of trusting an idle store', async () => {
  api.mockImplementation(async request =>
    request.path.endsWith('/jobs') ? { jobs: [{ kind: 'quickstart', status: 'running' }] } : eligible
  )
  await due()
  expect($activeTip.get()).toBeNull()
})
it('does not repeat a timed-out bubble for a week, then reads completion live', async () => {
  await due()
  await act(async () => {
    $activeTip.set(null)
  })
  await due()
  expect($activeTip.get()).toBeNull()
  vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 8 * 24 * 60 * 60_000)
  api.mockImplementation(async request =>
    request.path.endsWith('/jobs') ? { jobs: [] } : { ...eligible, update_available: false }
  )
  await due()
  expect($activeTip.get()).toBeNull()
  expect(api.mock.calls.filter(([request]) => request.path.endsWith('/status')).length).toBeGreaterThan(1)
})
it.each(['connection', 'profile', 'tips-off', 'job-started'])(
  'invalidates a visible action when %s changes',
  async context => {
    await due()
    const action = $activeTip.get()!.action!
    await act(async () => {
      if (context === 'connection') {
        $connection.set({ mode: 'remote' } as never)
      }

      if (context === 'profile') {
        $activeGatewayRoute.set('other')
      }

      if (context === 'tips-off') {
        $tipsEnabled.set(false)
      }

      if (context === 'job-started') {
        $localRuntimeJobs.set([{ kind: 'quickstart', status: 'running' }] as never)
      }

      action.onSelect()
    })
    expect(api.mock.calls.some(([request]) => request.method === 'POST')).toBe(false)
    expect(screen.getByRole('status').textContent).toBe('/')
    $activeGatewayRoute.set('default')
  }
)
it.each(['connection', 'profile'])('rejects a stale response after a %s round trip', async context => {
  let resolve!: (value: unknown) => void
  api.mockImplementation(request =>
    request.path.endsWith('/status')
      ? new Promise(done => {
          resolve = done
        })
      : Promise.resolve({ jobs: [] })
  )
  await act(async () => {
    offer()
  })

  if (context === 'connection') {
    const original = $connection.get()
    $connection.set({ mode: 'remote' } as never)
    $connection.set(original)
  } else {
    $activeGatewayRoute.set('other')
    $activeGatewayRoute.set('default')
  }

  await act(async () => {
    resolve(eligible)
  })
  await act(async () => {
    offer()
  })
  expect($activeTip.get()).toBeNull()
  await act(async () => {
    resolve({ ...eligible, update_available: false })
  })
  await act(async () => {
    offer()
  })
})
it.each(['remote', 'flag-off', 'tips-off', 'runtime-install', 'quickstart'])(
  'declines before reading when %s',
  async guard => {
    if (guard === 'remote') {
      $connection.set({ mode: 'remote' } as never)
    }

    if (guard === 'flag-off') {
      $localModelsEnabled.set(false)
    }

    if (guard === 'tips-off') {
      $tipsEnabled.set(false)
    }

    if (guard === 'runtime-install' || guard === 'quickstart') {
      $localRuntimeJobs.set([{ kind: guard, status: 'running' }] as never)
    }

    api.mockClear()
    await due()
    expect($activeTip.get()).toBeNull()
    expect(api).not.toHaveBeenCalled()
  }
)
it('dismisses per target build without installing and persists the retirement', async () => {
  await due()
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: en.tips.close }))
  })
  expect(JSON.parse(localStorage.getItem('hermes.desktop.tips.retired.v1')!)).toContain(
    'local-runtime-update:next-build'
  )
  await due()
  expect($activeTip.get()).toBeNull()
  api.mockImplementation(async request =>
    request.path.endsWith('/jobs') ? { jobs: [] } : { ...eligible, configured_tag: 'later-build' }
  )
  vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 61_000)
  await due()
  expect($activeTip.get()?.tipId).toBe('local-runtime-update:later-build')
  expect(api.mock.calls.some(([request]) => request.method === 'POST')).toBe(false)
})
it.each([{ enabled: false }, { runtime_installed: false }, { update_available: false }])(
  'does not offer when backend eligibility is missing: %j',
  async overrides => {
    api.mockImplementation(async request =>
      request.path.endsWith('/jobs') ? { jobs: [] } : { ...eligible, ...overrides }
    )
    await due()
    expect($activeTip.get()).toBeNull()
    expect(api.mock.calls.some(([request]) => request.method === 'POST')).toBe(false)
  }
)
it('the real Update now button navigates and sends exactly one install POST, never on show', async () => {
  await due()
  expect(api.mock.calls.filter(([request]) => request.method === 'POST')).toHaveLength(0)
  expect(screen.getByText(/active local requests may be interrupted/i)).toBeTruthy()
  const button = screen.getByRole('button', { name: 'Update now' })
  await act(async () => {
    fireEvent.click(button)
    fireEvent.click(button)
  })
  expect(screen.getByRole('status').textContent).toBe(`${SETTINGS_ROUTE}?tab=providers&pview=local`)
  expect(api.mock.calls.filter(([request]) => request.method === 'POST').map(([request]) => request)).toEqual([
    { body: { backend: null }, method: 'POST', path: '/api/local-models/runtime/install' }
  ])
  expect($activeTip.get()).toBeNull()
})
