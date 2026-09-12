import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import { $confirmRequest, settleConfirm } from '@/store/confirm'

const registry = atom({ connections: [{ id: 'local' }, { id: 'other', kind: 'remote' }] })
vi.mock('@/store/connections', () => ({
  $connectionsRegistry: registry,
  refreshConnectionsRegistry: async () => registry.get()
}))
vi.mock('@/store/session', () => ({ $connection: atom({ mode: 'local' }) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn(), dismissNotification: vi.fn() }))
const { runGatewayRestart } = await import('./system-actions')
const { applyBackendUpdate, applyEverythingUpdate } = await import('./updates')

afterEach(() => {
  settleConfirm(false)
  cleanup()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

it.each(['restart', 'update', 'cancel-restart', 'cancel-update', 'all', 'cancel-all'])(
  'requires a typed prompt before the real action controller dispatches: %s',
  async mode => {
    const calls: { path: string; body?: Record<string, unknown> }[] = []

    const updateAll = vi.fn().mockResolvedValue({ ok: true, results: [] })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      api: async (request: { path: string; body?: Record<string, unknown> }) => {
        calls.push(request)

        if (request.path === '/api/gateway/restart') {
          return { ok: true, name: 'gateway-restart', pid: 123 }
        }

        if (request.path === '/api/hermes/update') {
          return { ok: false, name: 'hermes-update', message: 'managed fixture', update_command: 'fixture' }
        }

        return { running: false, exit_code: 0, lines: [] }
      },
      connections: { updateAll },
      updates: { check: async () => ({ behind: 0, updateAvailable: false }) }
    }
    render(<ConfirmHost />)
    const restart = mode.endsWith('restart')
    const all = mode.endsWith('all')
    let pending!: Promise<unknown>
    act(() => {
      pending = all ? applyEverythingUpdate() : restart ? runGatewayRestart() : applyBackendUpdate()
    })

    if (mode === 'restart') {expect(runGatewayRestart()).toBe(pending)}

    try {
      const input = await screen.findByRole('textbox')
      expect(calls).toHaveLength(0)
      expect(updateAll).not.toHaveBeenCalled()

      if (mode.startsWith('cancel')) {
        fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
      } else {
        fireEvent.change(input, { target: { value: restart ? 'RESTART' : 'UPDATE' } })
        fireEvent.keyDown(input, { key: 'Enter' })
      }

      let result: unknown
      await act(async () => {
        result = await pending
      })

      if (restart) {expect(result).toBe(!mode.startsWith('cancel'))}

      if (mode.startsWith('cancel')) {
        expect(calls).toHaveLength(0)
        expect(updateAll).not.toHaveBeenCalled()
      } else {
        const body = all ? updateAll.mock.calls[0][0].mutation : calls[0].body
        expect(body).toMatchObject({ confirmation: restart ? 'RESTART' : 'UPDATE' })
        expect(body.idempotency_key).toMatch(/^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$/)
      }

      await waitFor(() => expect($confirmRequest.get()).toBeNull())
    } finally {
      settleConfirm(false)
      await pending
    }
  }
)
