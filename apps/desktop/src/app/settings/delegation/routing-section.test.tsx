import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DelegationRouting } from '@/api/delegation'
import type { HermesApiRequest } from '@/global'
import type { ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsOwner, $agentPluginsProfile, $agentPluginsStatus } from '@/store/agent-plugins'

import { DelegationRoutingSection } from './routing-section'
import { SubagentModelGate } from './subagent-model-gate'

const { lookupPluginCatalogEntry, openCatalogPluginInstall, requestGatewayForAgent } = vi.hoisted(() => ({
  lookupPluginCatalogEntry: vi.fn(),
  openCatalogPluginInstall: vi.fn(),
  requestGatewayForAgent: vi.fn()
}))

vi.mock('@/lib/plugin-catalog', async importOriginal => ({
  ...(await importOriginal<object>()),
  lookupPluginCatalogEntry: (...args: unknown[]) => lookupPluginCatalogEntry(...args)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<object>()),
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))

vi.mock('@/store/plugin-catalog-install', async importOriginal => ({
  ...(await importOriginal<object>()),
  openCatalogPluginInstall: (...args: unknown[]) => openCatalogPluginInstall(...args)
}))

// A stateful stand-in for the backend's /api/delegation routes (same shapes, same refusal).
let routing: DelegationRouting
const writes: HermesApiRequest[] = []

function backend(names: string[]): DelegationRouting['backends'] {
  return names.map(name => ({
    name,
    label: name,
    description: '',
    available: true,
    reason: '',
    builtin: name === 'default'
  }))
}

beforeEach(() => {
  queryClient.clear()
  writes.length = 0
  requestGatewayForAgent.mockReset()
  lookupPluginCatalogEntry.mockReset().mockResolvedValue({
    entry: { name: 'advanced-delegation', repo: 'https://github.com/example/advanced-delegation' },
    ok: true
  })
  openCatalogPluginInstall.mockReset()
  $agentPlugins.set([])
  $agentPluginsOwner.set(undefined)
  $agentPluginsProfile.set(undefined)
  $agentPluginsStatus.set('idle')
  routing = {
    configured: 'default',
    blocked_reason: null,
    backends: backend(['default']),
    model_map: { options: [], default: null },
    profile_routes: { profiles: [], default: null },
    profiles: ['default', 'work']
  }
  window.hermesDesktop = {
    ...window.hermesDesktop,
    api: async (request: HermesApiRequest) => {
      if (request.path === '/api/delegation/routing') {
        return structuredClone(routing)
      }

      if (request.path === '/api/delegation/backend') {
        writes.push(request)
        const name = (request.body as { backend: string }).backend

        if (!routing.backends.some(b => b.name === name)) {
          throw new Error(`400: Delegation backend '${name}' is not installed for this profile.`)
        }

        routing = { ...routing, configured: name, blocked_reason: null }

        return { ok: true, backend: name }
      }

      if (request.path.startsWith('/api/model/options')) {
        return { providers: [] }
      }

      throw new Error('404: Not Found')
    }
  } as typeof window.hermesDesktop
})

afterEach(cleanup)

function mount(scope: ProfileScope = 'work', ownerScope: ProfileScope = scope) {
  return render(
    <QueryClientProvider client={queryClient}>
      <DelegationRoutingSection ownerScope={ownerScope} scope={scope} />
      <SubagentModelGate scope={scope}>
        <div data-testid="subagent-model-row">Subagent Model</div>
      </SubagentModelGate>
    </QueryClientProvider>
  )
}

const radio = (name: RegExp) => screen.getByRole('radio', { name }) as HTMLButtonElement

describe('Subagents routing', () => {
  it('reloads Advanced Delegation for the exact gateway when two owners share a profile name', async () => {
    const plugin = {
      description: '',
      key: 'advanced-delegation',
      name: 'advanced-delegation',
      source: 'user',
      status: 'disabled',
      version: '1.0.0'
    }

    const sourceOwner = { baseUrl: 'https://foreground.example', mode: 'remote' as const, token: 'synthetic-source' }

    const owner = (connectionId: string, token: string) => ({
      cacheOwner: { baseUrl: `https://${connectionId}.example`, mode: 'remote' as const, token },
      connectionId,
      connectionOwner: sourceOwner,
      connectionOwnerProfile: 'foreground',
      profile: 'default'
    })

    const requestScope = ({ cacheOwner: _cacheOwner, ...scope }: ReturnType<typeof owner>): ProfileScope => scope

    requestGatewayForAgent.mockImplementation(async (connectionId, _profile, method, params) => {
      if (method === 'plugins.manage' && params?.action === 'list') {
        return { plugins: connectionId === 'gateway-a' ? [plugin] : [] }
      }

      throw new Error('Unexpected gateway request')
    })

    const gatewayA = owner('gateway-a', 'synthetic-token-a')
    const gatewayB = owner('gateway-b', 'synthetic-token-b')
    const view = mount(requestScope(gatewayA), gatewayA)

    await screen.findByText('Advanced Delegation is installed but turned off for this profile.')
    expect(screen.getByRole('button', { name: 'Turn on' })).toBeTruthy()

    view.rerender(
      <QueryClientProvider client={queryClient}>
        <DelegationRoutingSection ownerScope={gatewayB} scope={requestScope(gatewayB)} />
      </QueryClientProvider>
    )

    await screen.findByText('Model routes and Profile routes come from the Advanced Delegation plugin.')
    expect(screen.queryByRole('button', { name: 'Turn on' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Install plugin' })).toBeTruthy()
    expect(requestGatewayForAgent.mock.calls.map(([connectionId]) => connectionId)).toEqual(['gateway-a', 'gateway-b'])
    expect(requestGatewayForAgent.mock.calls.map(([, , , , , , options]) => options)).toEqual([
      expect.objectContaining({ expectedConnectionOwner: gatewayA.cacheOwner, expectedOwner: { profile: 'foreground', connectionOwner: sourceOwner } }),
      expect.objectContaining({ expectedConnectionOwner: gatewayB.cacheOwner, expectedOwner: { profile: 'foreground', connectionOwner: sourceOwner } })
    ])

    fireEvent.click(screen.getByRole('button', { name: 'Install plugin' }))
    await waitFor(() =>
      expect(openCatalogPluginInstall).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'advanced-delegation' }),
        'default',
        expect.objectContaining({ owner: gatewayB, ownerIdentity: expect.stringContaining('gateway-b') })
      )
    )
  })

  it('shows plugin modes with an install hint, never as choosable, until their backend registers', async () => {
    mount()
    await screen.findByText('Model routes and Profile routes come from the Advanced Delegation plugin.')

    expect(radio(/^Default/).getAttribute('aria-checked')).toBe('true')
    expect(radio(/^Model routes/).disabled).toBe(true)
    expect(radio(/^Profile routes/).disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Install plugin' })).toBeTruthy()
    expect(writes).toHaveLength(0)
  })

  it('choosing model routes makes the subagent model visibly inactive, and Switch to Default restores it', async () => {
    routing = {
      ...routing,
      backends: backend(['default', 'model_map', 'profile_routes']),
      model_map: {
        options: [{ name: 'fast', description: 'Quick', provider: 'openrouter', model: 'm' }],
        default: 'fast'
      }
    }
    mount()
    await waitFor(() => expect(radio(/^Model routes/).disabled).toBe(false))
    expect(screen.queryByText('Not used in this mode')).toBeNull()

    fireEvent.click(radio(/^Model routes/))

    await screen.findByText('Not used in this mode')
    expect(writes.map(w => [w.profile, (w.body as { backend: string }).backend])).toEqual([['work', 'model_map']])
    expect(screen.getByText(/Subagents that pick no route use “fast”/)).toBeTruthy()
    expect(screen.getByTestId('subagent-model-row').closest('[inert]')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Switch to Default' }))

    await waitFor(() => expect(screen.queryByText('Not used in this mode')).toBeNull())
    expect(screen.getByTestId('subagent-model-row').closest('[inert]')).toBeNull()
    expect(routing.configured).toBe('default')
  })

  it('keeps a saved mode that cannot run selected, says subagents will not start, and offers the fixes', async () => {
    routing = { ...routing, configured: 'model_map', blocked_reason: "Delegation backend 'model_map' is not installed" }
    mount()

    await screen.findByText('Subagents will not start until Advanced Delegation is installed.')
    expect(radio(/^Model routes/).getAttribute('aria-checked')).toBe('true')
    expect(radio(/^Default/).getAttribute('aria-checked')).toBe('false')
    expect(screen.getByText('Not running')).toBeTruthy()
    expect(screen.getAllByRole('button', { name: 'Install plugin' })).toHaveLength(1)
    // The subagent model does not apply in this mode either: it stays inert, never silently "back on".
    expect(screen.getByTestId('subagent-model-row').closest('[inert]')).not.toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Switch to Default' })[0])

    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
    expect(radio(/^Default/).getAttribute('aria-checked')).toBe('true')
    expect(routing.configured).toBe('default')
  })

  it('gates the subagent model in a route mode with no routes: unrouted tasks run on the main agent model', async () => {
    routing = { ...routing, configured: 'model_map', backends: backend(['default', 'model_map', 'profile_routes']) }
    mount()

    await screen.findByText(/Subagents that pick no route use the main agent's model/)
    expect(screen.getByTestId('subagent-model-row').closest('[inert]')).not.toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it("lists another plugin's backend by its own label and gates the subagent model while it is saved", async () => {
    routing = {
      ...routing,
      configured: 'swarm',
      backends: [
        ...backend(['default']),
        { name: 'swarm', label: 'Swarm', description: '', available: true, reason: '', builtin: false }
      ]
    }
    mount()

    await screen.findByText(/Swarm decides each subagent's model/)
    expect(radio(/^Swarm/).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByText('From a plugin. Its settings live with that plugin.')).toBeTruthy()
  })
})
