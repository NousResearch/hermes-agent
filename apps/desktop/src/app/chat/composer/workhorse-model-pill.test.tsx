import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $activeGatewayProfile } from '@/store/profile'

import { WorkhorseModelPill } from './workhorse-model-pill'

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  save: vi.fn(),
  cacheWrite: vi.fn(),
  previous: { delegation: {} as Record<string, unknown> },
  notifyError: vi.fn(),
  onSelectRef: { current: null as null | ((s: { provider: string; model: string }) => void) }
}))

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    saveHermesConfig: (config: Record<string, unknown>, profile?: unknown) => mocks.save(config, profile)
  }
})

vi.mock('@/app/hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: mocks.config }),
  hermesConfigCacheWriter: () => (patch: unknown) => {
    const fn = patch as (prev: unknown) => unknown
    mocks.cacheWrite(fn(mocks.previous))
  }
}))

vi.mock('@/components/model-picker', () => ({
  ModelPickerDialog: ({ onSelect }: { onSelect: (s: { provider: string; model: string }) => void }) => {
    mocks.onSelectRef.current = onSelect

    return <div data-testid="workhorse-picker-dialog" />
  }
}))

vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/store/notifications', () => ({
  notifyError: (...args: unknown[]) => mocks.notifyError(...args)
}))

afterEach(() => {
  cleanup()
  mocks.save.mockClear()
  mocks.cacheWrite.mockClear()
  mocks.notifyError.mockClear()
  mocks.config = {}
  $activeGatewayProfile.set('default')
})

beforeEach(() => {
  mocks.previous = { delegation: { model: '', provider: '' } }
})

function renderPill(over: Partial<{ compact: boolean; disabled: boolean }> = {}) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <WorkhorseModelPill disabled={false} {...over} />
    </I18nProvider>
  )
}

describe('WorkhorseModelPill', () => {
  it('shows the pinned delegation model', () => {
    mocks.config = { delegation: { model: 'deepseek/deepseek-v4-flash', provider: 'deepseek' } }
    renderPill()

    expect(screen.getByTestId('workhorse-model-pill')).toBeTruthy()
    expect(screen.getByText(/flash/i)).toBeTruthy()
  })

  it('shows an inherit label when no workhorse model is pinned', () => {
    mocks.config = {}
    renderPill()

    expect(screen.getByTestId('workhorse-model-pill')).toBeTruthy()
    expect(screen.getByText('inherit')).toBeTruthy()
  })

  it('opens the shared model picker dialog on click', async () => {
    mocks.config = {}
    renderPill()

    act(() => screen.getByTestId('workhorse-model-pill').click())

    expect(await screen.findByTestId('workhorse-picker-dialog')).toBeTruthy()
  })

  it('writes the delegation section on selection and mirrors the cache', async () => {
    mocks.config = { delegation: { provider: '', model: '' } }
    mocks.save.mockResolvedValueOnce({ ok: true })
    renderPill()

    act(() => screen.getByTestId('workhorse-model-pill').click())

    await act(async () => {
      mocks.onSelectRef.current?.({ provider: 'deepseek', model: 'deepseek/deepseek-v4-flash' })
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(mocks.save).toHaveBeenCalledWith(
      { delegation: { model: 'deepseek/deepseek-v4-flash', provider: 'deepseek' } },
      'default'
    )
    expect(mocks.cacheWrite).toHaveBeenCalledWith({
      delegation: { model: 'deepseek/deepseek-v4-flash', provider: 'deepseek' }
    })
  })

  it('surfaces a failed write without painting the cache', async () => {
    mocks.config = { delegation: { provider: '', model: '' } }
    mocks.save.mockRejectedValueOnce(new Error('boom'))
    renderPill()

    act(() => screen.getByTestId('workhorse-model-pill').click())

    await act(async () => {
      mocks.onSelectRef.current?.({ provider: 'deepseek', model: 'deepseek/deepseek-v4-flash' })
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(mocks.notifyError).toHaveBeenCalled()
    expect(mocks.cacheWrite).not.toHaveBeenCalled()
  })

  it('merges the selection into the latest cached delegation state', async () => {
    mocks.config = { delegation: { model: 'old', provider: 'old-provider' } }
    mocks.previous = { delegation: { model: 'latest', provider: 'latest-provider', reasoning_effort: 'high' } }
    mocks.save.mockResolvedValueOnce({ ok: true })
    renderPill()

    await act(async () => {
      mocks.onSelectRef.current?.({ provider: 'new-provider', model: 'new-model' })
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(mocks.cacheWrite).toHaveBeenCalledWith({
      delegation: { model: 'new-model', provider: 'new-provider', reasoning_effort: 'high' }
    })
  })
})
