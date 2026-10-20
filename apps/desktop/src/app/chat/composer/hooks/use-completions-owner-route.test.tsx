import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type * as GatewayStore from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { useAtCompletions } from './use-at-completions'
import { useSlashCompletions } from './use-slash-completions'

const routed = vi.hoisted(() => ({
  requestGatewayForAgent: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForAgent: routed.requestGatewayForAgent
}))

const SESSION = 'sess-research-1'
const OWNER = { connectionId: 'local', profile: 'research' }

const CATALOG = {
  categories: [{ name: 'Session', pairs: [['/new', 'Start a new session']] }],
  pairs: [['/new', 'Start a new session']]
}

// The ambient (launch-profile) socket of a multiplexed local gateway: its
// authority refuses another profile's session_id, exactly like
// gateway/session_discovery.py and gateway/session_complete_path.py do.
function ambientGateway() {
  return {
    request: vi.fn(async () => {
      throw new Error('not_found')
    })
  } as unknown as HermesGateway & { request: ReturnType<typeof vi.fn> }
}

function ownerAnswers(method: string, params: Record<string, unknown>) {
  if (method === 'commands.catalog') {
    return CATALOG
  }

  if (method === 'complete.slash') {
    return { items: [{ text: '/research-only', display: '/research-only' }], replace_from: 1 }
  }

  return { items: [{ text: `@file:${String(params.word).slice(1)}notes.md`, display: 'notes.md' }] }
}

type Search = (query: string) => readonly Unstable_TriggerItem[]

/** Read through `api` each time: the adapter re-renders with its results. */
async function settle(api: { search?: Search }, query: string) {
  await act(async () => {
    api.search!(query)
    await Promise.resolve()
  })
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 150))
  })

  return api.search!(query)
}

beforeEach(() => {
  routed.requestGatewayForAgent.mockImplementation(
    async (_connectionId: string, _profile: string, method: string, params: Record<string, unknown>) =>
      ownerAnswers(method, params)
  )
  setSessionOwnerHint(SESSION, OWNER)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  _resetSessionOwnerHintsForTests()
  routed.requestGatewayForAgent.mockReset()
})

// #141583: a non-primary-profile window's composer sent completion RPCs on the
// window's ambient socket, whose authority is the launch profile's — every `/`
// and `@` lookup for another profile's session came back not_found and the
// palette said "No matches". Completions must go to the session's owner.
describe('composer completions route to the session owner', () => {
  it('serves / catalog and typed lookups from the owning profile, not the ambient socket', async () => {
    const gateway = ambientGateway()
    const api: { search?: Search } = {}

    function Probe() {
      api.search = useSlashCompletions({ gateway, sessionId: SESSION }).adapter.search

      return null
    }

    render(<Probe />)

    const bare = await settle(api, '')
    const typed = await settle(api, 'res')

    expect(bare.map(item => (item.metadata as { command?: string }).command)).toContain('/new')
    expect(typed.map(item => (item.metadata as { command?: string }).command)).toContain('/research-only')
    expect(gateway.request).not.toHaveBeenCalled()
    expect(routed.requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'commands.catalog', {
      session_id: SESSION
    })
    expect(routed.requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'complete.slash', {
      session_id: SESSION,
      text: '/res'
    })
  })

  it('serves @ path completions from the owning profile, not the ambient socket', async () => {
    const gateway = ambientGateway()
    const api: { search?: Search } = {}

    function Probe() {
      api.search = useAtCompletions({ cwd: '/repo', gateway, sessionId: SESSION }).adapter.search

      return null
    }

    render(<Probe />)

    const items = await settle(api, 'no')

    expect(items.map(item => (item.metadata as { rawText?: string }).rawText)).toContain('@file:nonotes.md')
    expect(gateway.request).not.toHaveBeenCalled()
    expect(routed.requestGatewayForAgent).toHaveBeenCalledWith('local', 'research', 'complete.path', {
      cwd: '/repo',
      session_id: SESSION,
      word: '@no'
    })
  })
})
