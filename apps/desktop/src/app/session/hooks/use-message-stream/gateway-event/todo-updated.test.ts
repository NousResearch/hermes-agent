import { describe, expect, it, vi } from 'vitest'

const probes = vi.hoisted(() => ({ setSnapshot: vi.fn() }))

vi.mock('@/store/todos', () => ({ setSessionTodoSnapshot: probes.setSnapshot }))
vi.mock('@/lib/slash-completion-cache', () => ({ invalidateSlashCompletions: vi.fn() }))
vi.mock('@/store/composer-status', () => ({ refreshBackgroundProcesses: vi.fn() }))
vi.mock('@/store/pet', () => ({ flashPetActivity: vi.fn(), setPetActivity: vi.fn() }))
vi.mock('@/store/subagents', () => ({ pruneDelegateFallbackSubagents: vi.fn(), upsertSubagent: vi.fn() }))
vi.mock('@/store/suggestion-providers/repair', () => ({ reportMcpToolResult: vi.fn() }))
vi.mock('@/store/suggestion-providers/skill', () => ({ invalidateSkillSuggestionIndex: vi.fn() }))
vi.mock('@/store/tool-diffs', () => ({ recordToolDiff: vi.fn() }))
vi.mock('@/store/tool-drafting', () => ({ setSessionDraftingTool: vi.fn() }))
vi.mock('@/store/workspace-events', () => ({
  notifyWorkspaceChanged: vi.fn(),
  toolChangedPath: vi.fn(),
  toolMayMutateFiles: vi.fn(() => false)
}))
vi.mock('../utils', () => ({ SUBAGENT_EVENT_TYPES: new Set(), toTodoPayload: (value: unknown) => value }))

const { handleToolEvent } = await import('./tools')

const snapshot = {
  generation: 3,
  revision: 2,
  session_id: 's1',
  todos: [{ content: 'Task', id: 't1', status: 'in_progress' }]
}

describe('todo.updated gateway event', () => {
  it('updates the authoritative cache without creating a transcript tool part', () => {
    probes.setSnapshot.mockClear()
    const upsertToolCall = vi.fn()

    const consumed = handleToolEvent({
      deps: {
        flushQueuedDeltas: vi.fn(),
        nativeSubagentSessionsRef: { current: new Set() },
        sessionInterrupted: vi.fn(() => false),
        updateSessionState: vi.fn(),
        upsertToolCall
      },
      event: { type: 'todo.updated' },
      explicitSid: 's1',
      fromActiveSource: () => true,
      isActiveEvent: true,
      occurredAt: 1,
      payload: snapshot,
      scheduleConfigRefresh: vi.fn(),
      sessionId: 's1'
    } as never)

    expect(consumed).toBe(true)
    expect(probes.setSnapshot).toHaveBeenCalledWith(snapshot)
    expect(upsertToolCall).not.toHaveBeenCalled()
  })
})
