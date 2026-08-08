import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $approvalModes } from '@/store/approval-mode'

import { ComposerApprovalMode } from './approval-mode'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
})

afterEach(() => {
  cleanup()
  $approvalModes.set({})
})

function renderApprovalMode(requestGateway = vi.fn(async (_method, params) => ({ value: params?.value ?? 'smart' }))) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ComposerApprovalMode profile="default" requestGateway={requestGateway} />
    </I18nProvider>
  )
}

describe('ComposerApprovalMode', () => {
  it("uses the status bar's shared approval state without issuing another config read", async () => {
    const requestGateway = vi.fn(async (_method, params) => ({ value: params?.value ?? 'smart' }))
    renderApprovalMode(requestGateway)

    await Promise.resolve()

    expect(requestGateway).not.toHaveBeenCalledWith('config.get', { key: 'approvals.mode' })
  })

  it('renders the active approval mode beside the composer and exposes every choice upward', async () => {
    renderApprovalMode()

    const trigger = screen.getByRole('button', { name: 'Approval mode: Smart' })
    expect(trigger.textContent).toContain('Smart')
    expect(trigger.querySelector('svg')?.getAttribute('class')).toContain('text-emerald-600')

    fireEvent.pointerDown(trigger, { button: 0 })

    expect(await screen.findByRole('menuitemradio', { name: /Manual/ })).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: /Smart/ })).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: /Full access/ })).toBeTruthy()
  })

  it('writes the new mode through the same profile-scoped gateway setting', async () => {
    const requestGateway = vi.fn(async (_method, params) => ({ value: params?.value ?? 'smart' }))
    renderApprovalMode(requestGateway)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Approval mode: Smart' }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /Full access/ }))

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.set', { key: 'approvals.mode', value: 'off' })
      const trigger = screen.getByRole('button', { name: 'Approval mode: Full access' })
      expect(trigger.className).toContain('rounded-[var(--radius-sm)]')
      expect(trigger.className).not.toContain('bg-amber-500/10')
      expect(trigger.querySelectorAll('svg')).toHaveLength(2)
      expect(trigger.querySelector('svg')?.getAttribute('class')).toContain('text-amber-600')
    })
  })
})
