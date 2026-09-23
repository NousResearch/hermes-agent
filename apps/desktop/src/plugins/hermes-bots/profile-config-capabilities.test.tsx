/**
 * The bot editor's Advanced section renders the REAL core Capabilities
 * surfaces — CapabilitiesView (installed skills + hub installs + detail),
 * ToolsetConfigPanel (per-toolset env/keys/model/post-setup) and ConnectorsTab
 * (per-server enable + OAuth + API keys) — pinned to the bot's own profile,
 * instead of bare checkbox stand-ins.
 *
 * All three are optional SDK namespace exports (hermes-agent#87317), so every
 * use site is feature-detected and older desktop builds keep the staged
 * checklist UI. The sharp edge is a REMOTE bot on a build whose CapabilitiesView
 * predates `supportsFixedConnection`: rendering the live surface there would
 * read and write the ACTIVE gateway's skills under the remote bot's name —
 * the wrong machine — so those builds must fail closed to "staged only".
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'
import type { CapabilityEntry } from './profile-config'
import type { RosterRow } from './types'

interface StubProps {
  fixedConnection?: string
  fixedProfile?: string
  gateway?: string
  profile?: unknown
  toolset?: string
}

/** The optional SDK exports, swapped per test to model each desktop build. */
const sdk = vi.hoisted(() => {
  const seen: Record<string, StubProps[]> = { ConnectorsTab: [], CapabilitiesView: [], ToolsetConfigPanel: [] }

  const spy = (name: string) => {
    const Stub = (props: StubProps) => {
      seen[name].push(props)

      return null
    }

    return Stub
  }

  return {
    exports: {} as Record<string, unknown>,
    gatewayReads: { count: 0 },
    request: vi.fn(async () => ({})),
    requestProfile: vi.fn(async () => ({})),
    seen,
    spy
  }
})

// The optional exports are read at MODULE scope, so each build's set has to
// be live at the moment `vi.resetModules()` re-evaluates the editor — a value
// baked into the factory would freeze the first test's build for the rest.
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  const mocked: Record<string, unknown> = {
    ...original,
    host: {
      ...original.host,
      getGateway: () => {
        sdk.gatewayReads.count += 1

        return 'ambient-gateway'
      },
      request: sdk.request,
      requestProfile: sdk.requestProfile
    },
    // The plugin bundle normally lands via `ctx.i18n.register` at load.
    usePluginI18n: () => translateBots
  }

  for (const name of ['ConnectorsTab', 'CapabilitiesView', 'ToolsetConfigPanel']) {
    Object.defineProperty(mocked, name, { configurable: true, enumerable: true, get: () => sdk.exports[name] })
  }

  return mocked
})

// Each build re-imports the editor's whole module graph, which outruns the
// default per-test budget on a cold transform cache.
vi.setConfig({ testTimeout: 30_000 })

const remoteBot: RosterRow = {
  connectionId: 'remote-a',
  name: 'default',
  remoteSource: true,
  route: { connectionId: 'remote-a', mode: 'remote', profile: 'default', targetProfile: 'backend-default' },
  sourceScoped: true,
  targetProfile: 'backend-default'
}

const localBot: RosterRow = {
  connectionId: 'local',
  name: 'default',
  remoteSource: false,
  route: { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
  sourceScoped: true,
  targetProfile: 'default'
}

const advancedState: {
  dirtyMcp: boolean
  dirtyModel: boolean
  dirtySkills: boolean
  dirtySoul: boolean
  dirtyToolsets: boolean
  loaded: boolean
  mcp: CapabilityEntry[]
  model: string
  provider: string
  skills: CapabilityEntry[]
  soul: string
  toolsets: CapabilityEntry[]
} = {
  dirtyMcp: false,
  dirtyModel: false,
  dirtySkills: false,
  dirtySoul: false,
  dirtyToolsets: false,
  loaded: true,
  mcp: [{ enabled: false, fromCatalog: true, installed: false, name: 'remote-mcp', requires: ['TOKEN'] }],
  model: '',
  provider: '',
  skills: [],
  soul: '',
  toolsets: []
}

function withQueryClient(children: ReactNode) {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {children}
    </QueryClientProvider>
  )
}

/** Load the editor against one desktop build's export set. */
async function renderEditor(
  exports: Record<string, unknown>,
  bot: RosterRow,
  state: Partial<typeof advancedState> = {}
) {
  sdk.exports = exports
  vi.resetModules()

  const { AdvancedProfileConfig } = await import('./profile-config')

  return render(
    withQueryClient(
      <AdvancedProfileConfig bot={bot} setState={() => undefined} state={{ ...advancedState, ...state }} />
    )
  )
}

beforeAll(async () => {
  // Warm the transform cache so the first test isn't charged for it.
  await import('./profile-config')
}, 120_000)

beforeEach(() => {
  vi.clearAllMocks()
  sdk.gatewayReads.count = 0

  for (const key of Object.keys(sdk.seen)) {
    sdk.seen[key].length = 0
  }
})

afterEach(() => {
  cleanup()
})

describe('a build whose CapabilitiesView cannot route connections', () => {
  const oldBuild = () => ({
    ConnectorsTab: sdk.spy('ConnectorsTab'),
    CapabilitiesView: sdk.spy('CapabilitiesView'),
    ToolsetConfigPanel: sdk.spy('ToolsetConfigPanel')
  })

  it('fails closed for a REMOTE bot \u2014 staged model + SOUL only', async () => {
    const { container } = await renderEditor(oldBuild(), remoteBot)

    // The model catalog settles first so the picker is past its spinner.
    expect(await screen.findByText('Provider')).toBeTruthy()
    expect(screen.getByText(/Remote capabilities require a newer desktop/)).toBeTruthy()
    expect(sdk.seen.CapabilitiesView).toHaveLength(0)
    expect(sdk.seen.ConnectorsTab).toHaveLength(0)
    expect(sdk.seen.ToolsetConfigPanel).toHaveLength(0)
    expect(screen.queryByText('Skills Hub')).toBeNull()
    expect(screen.queryByRole('button', { name: /Set up/ })).toBeNull()
    // Model + SOUL still edit, and stay staged until the user saves.
    expect(container.querySelector('textarea')).toBeTruthy()
    // Nothing may read the AMBIENT gateway on behalf of a remote bot.
    expect(sdk.gatewayReads.count).toBe(0)

    container.querySelectorAll('button, input').forEach(node => (node as HTMLElement).click())

    expect(sdk.request).not.toHaveBeenCalled()
    expect(sdk.gatewayReads.count).toBe(0)
  })

  it('keeps the LOCAL capability surfaces, each scoped to the bot profile', async () => {
    await renderEditor(oldBuild(), localBot, {
      skills: [{ enabled: true, name: 'staged-skill' }],
      toolsets: [{ enabled: true, name: 'local-tools' }]
    })

    // The staged checklist and the hub search section both survive here.
    expect(screen.getByText('staged-skill')).toBeTruthy()
    expect(screen.getByText('Skills Hub')).toBeTruthy()
    expect(sdk.seen.ToolsetConfigPanel[0]).toEqual({
      profile: { connectionId: 'local', profile: 'default' },
      toolset: 'local-tools'
    })
    expect(sdk.seen.ConnectorsTab[0]).toEqual({
      gateway: 'ambient-gateway',
      profile: { connectionId: 'local', profile: 'default' }
    })
    expect(sdk.gatewayReads.count).toBe(1)
  })
})

describe('a connection-aware CapabilitiesView', () => {
  it('receives the connection and the BACKEND profile separately', async () => {
    const Routed = sdk.spy('CapabilitiesView')

    ;(Routed as unknown as { supportsFixedConnection: boolean }).supportsFixedConnection = true

    await renderEditor({ CapabilitiesView: Routed }, remoteBot)

    // The alias's logical name is `default`; the backend row it activates into
    // is `backend-default`. Pinning the logical name would edit the wrong one.
    expect(sdk.seen.CapabilitiesView[0]).toMatchObject({
      fixedConnection: 'remote-a',
      fixedProfile: 'backend-default'
    })
  })

  it('#93492: degrades an orphaned row to its own name instead of throwing', async () => {
    const Routed = sdk.spy('CapabilitiesView')

    ;(Routed as unknown as { supportsFixedConnection: boolean }).supportsFixedConnection = true

    // A row whose owning connection was removed resolves to no route at all.
    // The editor renders on the render path, so a throw here would take the
    // whole pane down through the dialog's error boundary.
    await renderEditor({ CapabilitiesView: Routed }, { ...remoteBot, connectionId: '', route: undefined })

    expect(sdk.seen.CapabilitiesView[0]).toEqual({ embedded: true, fixedProfile: 'default' })
  })
})

describe('a build with no Capabilities exports at all', () => {
  const bareBuild = { ConnectorsTab: undefined, CapabilitiesView: undefined, ToolsetConfigPanel: undefined }

  it('keeps the checkbox MCP list with its inline setup button', async () => {
    await renderEditor(bareBuild, localBot)

    expect(screen.getByText('remote-mcp')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Set up/ })).toBeTruthy()
  })
})
