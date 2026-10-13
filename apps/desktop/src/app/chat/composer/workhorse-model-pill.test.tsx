import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { WorkhorseModelPill } from './workhorse-model-pill'

const mocks = vi.hoisted(() => ({
  delegation: {} as Record<string, unknown>,
  ready: true,
  commit: vi.fn()
}))

vi.mock('./workhorse-config', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkhorseConfig: () => ({ ...mocks, profile: 'worker', connectionId: 'connection-a' })
}))
vi.mock('@/components/model-picker', () => ({
  ModelPickerDialog: ({ open, onReset, resetLabel }: { open: boolean; onReset: () => void; resetLabel: string }) =>
    open ? <button onClick={onReset}>{resetLabel}</button> : null
}))
vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

beforeEach(() => {
  mocks.delegation = {}
  mocks.ready = true
  mocks.commit.mockReset()
})
afterEach(cleanup)

function renderPill(over: Partial<{ compact: boolean; disabled: boolean }> = {}) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <WorkhorseModelPill disabled={false} {...over} />
    </I18nProvider>
  )
}

describe('WorkhorseModelPill presentation', () => {
  it('identifies a pinned model as a subagent model', () => {
    mocks.delegation = { model: 'deepseek/deepseek-v4-flash', provider: 'deepseek' }
    renderPill()
    expect(screen.getByTestId('workhorse-model-pill').textContent).toContain('Subagents')
    expect(screen.getByText(/flash/i)).toBeTruthy()
  })
  it('shows inheritance when no model is pinned', () => {
    renderPill()
    expect(screen.getByText('inherit')).toBeTruthy()
  })
  it('keeps the subagent label visible in compact mode', () => {
    renderPill({ compact: true })
    expect(screen.getByText('Subagents')).toBeTruthy()
  })
  it('opens the shared picker and exposes the independent model reset', () => {
    renderPill()
    fireEvent.click(screen.getByTestId('workhorse-model-pill'))
    fireEvent.click(screen.getByRole('button', { name: 'Inherit parent model and route' }))
    expect(mocks.commit).toHaveBeenCalledWith('model', {
      model: '',
      provider: '',
      base_url: '',
      api_key: '',
      api_mode: ''
    })
  })
  it('disables writes until a config read has resolved', () => {
    mocks.ready = false
    renderPill()
    expect(screen.getByTestId('workhorse-model-pill').hasAttribute('disabled')).toBe(true)
  })
  it('honors the composer disabled state', () => {
    renderPill({ disabled: true })
    expect(screen.getByTestId('workhorse-model-pill').hasAttribute('disabled')).toBe(true)
  })
})
