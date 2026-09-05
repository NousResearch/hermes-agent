import { describe, expect, it } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { GatewayEventPayload } from '@/lib/chat-messages'
import type { SessionRuntimeInfo } from '@/types/hermes'

import { applySessionInfoStatePatch, sessionInfoStatePatch } from './use-message-stream/utils'
import { applyRuntimeInfo } from './use-session-actions/utils'

const binding = { requestId: 'a', projectId: 'project-a', sourcePath: '/repo', cwd: '/repo/.worktrees/a', repoRoot: '/repo', branch: 'task/a' }

describe('coding workspace runtime metadata', () => {
  it('hydrates create/resume and event state without foreground publication; heartbeats keep identity', () => {
    const info = { coding_workspace: binding }
    const resumed = applyRuntimeInfo(info as SessionRuntimeInfo, { foreground: false })
    expect(resumed).toEqual({ codingWorkspace: binding })
    const patch = sessionInfoStatePatch(info as GatewayEventPayload)
    const initial = { cwd: '/repo/.worktrees/a', branch: 'task/a' } as ClientSessionState
    const updated = applySessionInfoStatePatch(initial, patch)
    expect(updated).toMatchObject({ codingWorkspace: binding })
    expect(applySessionInfoStatePatch(updated, sessionInfoStatePatch(JSON.parse(JSON.stringify(info))))).toBe(updated)
    expect(applyRuntimeInfo({ coding_workspace: null } as SessionRuntimeInfo, { foreground: false })).toEqual({ codingWorkspace: null })
    expect(applySessionInfoStatePatch(updated, sessionInfoStatePatch({ coding_workspace: null } as GatewayEventPayload))).toMatchObject({ codingWorkspace: null })
    expect(sessionInfoStatePatch({})).toEqual({})
  })
})
