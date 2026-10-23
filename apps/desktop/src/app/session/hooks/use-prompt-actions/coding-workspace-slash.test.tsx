import { cleanup } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { $codingWorkspaceDrafts, codingWorkspaceKey, setCodingWorkspaceIntent } from '@/store/coding-workspaces'
import { requestGatewayForAgent } from '@/store/gateway'
import { $newChatRoute } from '@/store/profile'
import { $activeSessionId, $connection, $sessions } from '@/store/session'

import { actRender, Harness, type HarnessHandle } from './prompt-actions-harness'

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()), requestGatewayForAgent: vi.fn()
}))
afterEach(() => {
  cleanup(); $codingWorkspaceDrafts.set({}); $newChatRoute.set(null); $activeSessionId.set(null)
})

test('a slash command captures its draft base before creating the workspace session', async () => {
  const owner = { connectionId: 'fixture-gateway', profile: 'coder', draftKey: 'fixture-draft' }
  $newChatRoute.set(owner); $sessions.set([]); $connection.set(null)
  setCodingWorkspaceIntent(owner, { path: '/source', mode: 'worktree' })
  vi.mocked(requestGatewayForAgent).mockResolvedValue({ items: [], sourceCwd: '/original' })
  const active = { current: null as string | null }

  const created = vi.fn(async () => {
    expect($codingWorkspaceDrafts.get()[codingWorkspaceKey(owner)].referenceCwd).toBe('/original')
    active.current = 'fixture-runtime'

    return active.current
  })

  const request = vi.fn(async () => ({ type: 'exec', lines: ['fixture slash output'] }) as never)
  let handle!: HarnessHandle
  await actRender(<Harness activeSessionId={null} activeSessionIdRef={active} createBackendSessionForSend={created}
    onReady={value => { handle = value }} refreshSessions={async () => undefined}
    requestGateway={request} storedSessionId={null} />)
  expect(await handle.submitText('/fixture-workspace', { composerScope: owner.draftKey })).toBe(true)
  expect(created).toHaveBeenCalledTimes(1)
  expect(created).toHaveBeenCalledWith('/fixture-workspace', { onComposerScopeAssigned: undefined }, owner.draftKey)
})
