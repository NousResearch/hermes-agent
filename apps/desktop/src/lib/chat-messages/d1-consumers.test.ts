import { fromThreadMessageLike, getAutoStatus } from '@assistant-ui/core/internal'
import { describe, expect, it } from 'vitest'
import { buildToolView } from '@/components/assistant-ui/tool/fallback-model'
import { deriveChangedFiles } from '@/components/assistant-ui/thread/changed-files'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { sealOpenToolParts, upsertToolPart } from '@/lib/chat-messages'
import { todosFromMessageContent } from '@/lib/todos'

const normalize = (parts: ReturnType<typeof upsertToolPart>) => fromThreadMessageLike(toRuntimeMessage({ id: 'd1', role: 'assistant', parts }), 'd1', getAutoStatus(false, false, false, false, undefined)).content

describe('D1 settlement and presentation consumers', () => {
  it('seals unknown outcomes without fabricating success and remains idempotent', () => {
    const messages = [{ id: 'd1', role: 'assistant' as const, parts: upsertToolPart([], { name: 'terminal', tool_id: 'missing' }, 'running', 1) }]
    const sealed = sealOpenToolParts(messages)
    const part = sealed[0].parts[0]
    expect('result' in part ? part.result : undefined).toBeUndefined()
    expect(part.completedAt).toBeDefined()
    expect(buildToolView(part as never, '').status).toBe('warning')
    expect(sealOpenToolParts(sealed)).toBe(sealed)
  })
  it('keeps side-channel diffs in the changed-files consumer after normalization', () => {
    const parts = upsertToolPart([], { name: 'patch', tool_id: 'patch', args: { path: '/tmp/a.ts' }, result: 'ok', inline_diff: '--- a/a.ts\n+++ b/a.ts\n@@ -1 +1 @@\n-old\n+new' }, 'complete', 2)
    expect(deriveChangedFiles(normalize(parts))).toEqual([{ path: '/tmp/a.ts', name: 'a.ts', added: 1, removed: 1 }])
  })
  it('keeps envelope-only todo completion and explicit clearing after normalization', () => {
    const todos = [{ id: 'a', content: 'Do it', status: 'completed' }]
    const parts = upsertToolPart([], { name: 'todo_list', tool_id: 'todos', result: 'ok', todos }, 'complete', 2)
    expect(todosFromMessageContent(normalize(parts))).toEqual(todos)
    const cleared = upsertToolPart(parts, { name: 'todo_list', tool_id: 'todos', todos: [] }, 'complete', 3)
    expect(todosFromMessageContent(normalize(cleared))).toEqual([])
  })
})
