import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

const { requestGateway } = vi.hoisted(() => ({ requestGateway: vi.fn() }))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({ useGatewayRequest: () => ({ requestGateway }) }))

import { $pluginInstallRequest } from '@/store/plugin-install-request'

import { PluginInstallModal } from './plugin-install-modal'

afterEach(() => {
  cleanup()
  $pluginInstallRequest.set(null)
  vi.unstubAllGlobals()
})

it('offers setup review after files install instead of reporting enabled or requiring a reclone', async () => {
  vi.stubGlobal('hermesDesktop', {
    probePluginRepo: vi.fn().mockResolvedValue({ ok: true, agent: true, desktop: false, warnings: [] })
  })
  requestGateway.mockImplementation(async (_method, params) => {
    if (params.action === 'list') {
      return { plugins: [] }
    }

    throw Object.assign(new Error('Review setup'), {
      data: { status: 'consent_required', installed: true, plugin_name: 'native-fixture', error: 'Review setup' }
    })
  })
  $pluginInstallRequest.set({ repo: 'owner/native-fixture' })
  render(
    <MemoryRouter initialEntries={['/new']}>
      <PluginInstallModal />
    </MemoryRouter>
  )
  const install = await screen.findByRole('button', { name: 'Install' })
  await waitFor(() => expect((install as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(install)
  const review = await screen.findByRole('button', { name: 'Review setup in Plugins' })
  expect(screen.getByText(/Files installed; enablement was not changed/)).toBeTruthy()
  fireEvent.click(review)
  await waitFor(() => expect($pluginInstallRequest.get()).toBeNull())
  expect(requestGateway.mock.calls.filter(([, p]) => p.action === 'install')).toHaveLength(1)
})
