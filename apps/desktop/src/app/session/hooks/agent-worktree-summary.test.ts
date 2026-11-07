import { describe, expect, it } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { GatewayEventPayload } from '@/lib/chat-messages'
import type { SessionRuntimeInfo } from '@/types/hermes'

import { applySessionInfoStatePatch, sessionInfoStatePatch } from './use-message-stream/utils'
import { applyRuntimeInfo } from './use-session-actions/utils'

const worktree = { cwd: '/repo/.worktrees/task-a', branch: 'task/a', repoRoot: '/repo', projectName: 'repo' }

describe('agent worktree runtime metadata', () => {
  it('hydrates create/resume and event state; identical heartbeats keep the state object', () => {
    const info = { agent_worktree: worktree }
    expect(applyRuntimeInfo(info as SessionRuntimeInfo, { foreground: false })).toEqual({ agentWorktree: worktree })
    const patch = sessionInfoStatePatch(info as GatewayEventPayload)
    const initial = { cwd: '/home/person/projects', branch: '' } as ClientSessionState
    const updated = applySessionInfoStatePatch(initial, patch)
    expect(updated).toMatchObject({ agentWorktree: worktree })
    // The session's own cwd is untouched: the badge is additive, never a re-home.
    expect(updated.cwd).toBe('/home/person/projects')
    expect(applySessionInfoStatePatch(updated, sessionInfoStatePatch(JSON.parse(JSON.stringify(info))))).toBe(updated)
    // A cleared badge (tree removed / session re-homed onto it) is an authoritative null, not "no opinion".
    expect(applyRuntimeInfo({ agent_worktree: null } as SessionRuntimeInfo, { foreground: false })).toEqual({
      agentWorktree: null
    })
    expect(
      applySessionInfoStatePatch(updated, sessionInfoStatePatch({ agent_worktree: null } as GatewayEventPayload))
    ).toMatchObject({ agentWorktree: null })
    // Absent means untouched.
    expect(
      applySessionInfoStatePatch(
        updated,
        sessionInfoStatePatch({ cwd: '/home/person/projects' } as GatewayEventPayload)
      )
    ).toBe(updated)
  })
})
