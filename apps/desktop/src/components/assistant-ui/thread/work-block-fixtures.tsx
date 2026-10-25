// Shared fixtures for the work block render tests ("Group work between replies").
import { fromThreadMessageLike } from '@assistant-ui/core/internal'
import { type ThreadMessage } from '@assistant-ui/react'
import { waitFor } from '@testing-library/react'
import { vi } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

export function stubWorkBlockEnvironment() {
  stubThreadEnvironment()
  stubThreadViewportSize()
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
}

const createdAt = new Date('2026-06-03T00:00:00.000Z')
export const at = (s: number) => createdAt.getTime() / 1000 + s

export const RUNNING = { type: 'running' } as ThreadMessage['status']

// Through fromThreadMessageLike, like the app's runtime repository: it drops
// blank text and reasoning, so fixtures see the parts production renders.
export function turn(
  content: unknown[],
  status: ThreadMessage['status'] = { type: 'complete', reason: 'stop' }
): ThreadMessage {
  return fromThreadMessageLike(
    { id: 'assistant-work', role: 'assistant', content, status, createdAt } as never,
    'assistant-work',
    status as never
  )
}

export const read = (id: string, path: string) => ({
  type: 'tool-call',
  toolCallId: id,
  toolName: 'read_file',
  args: { path },
  argsText: JSON.stringify({ path }),
  timestamp: at(1),
  completedAt: at(2),
  result: { content: 'x' }
})

export const term = (id: string, command = 'ls') => ({
  type: 'tool-call',
  toolCallId: id,
  toolName: 'terminal',
  args: { command },
  argsText: JSON.stringify({ command }),
  timestamp: at(3),
  completedAt: at(4),
  result: { exit_code: 0, stdout: 'ok' }
})

export const search = (id: string, query: string) => ({
  type: 'tool-call',
  toolCallId: id,
  toolName: 'web_search',
  args: { query },
  argsText: JSON.stringify({ query }),
  timestamp: at(3),
  completedAt: at(4),
  result: { results: [] }
})

export const edit = (id: string, path = '/repo/a.ts') => ({
  type: 'tool-call',
  toolCallId: id,
  toolName: 'patch',
  args: { path, old_string: 'a', new_string: 'b' },
  argsText: '{}',
  timestamp: at(5),
  completedAt: at(6),
  result: { success: true, diff: `--- a${path}\n+++ b${path}\n@@ -1 +1 @@\n-beforeEdit\n+afterEdit\n` }
})

export const refusedWrite = (id: string) => ({
  type: 'tool-call',
  toolCallId: id,
  toolName: 'write_file',
  args: { path: '/repo/b.ts', content: 'x' },
  argsText: '{}',
  timestamp: at(5),
  completedAt: at(6),
  result: { success: false, error: 'Refusing to overwrite /repo/b.ts' }
})

/** A call still waiting on its result. */
export const pending = <T extends object>(part: T) => ({ ...part, result: undefined, completedAt: undefined })

export const Harness = ({ message }: { message: ThreadMessage }) => (
  <ThreadRuntime messages={[message]}>
    <Thread />
  </ThreadRuntime>
)

export async function findHeader(container: HTMLElement, index = 0): Promise<HTMLElement> {
  return waitFor(() => {
    const node = container.querySelectorAll<HTMLElement>('[data-work-block-header]')[index]

    if (!node) {
      throw new Error('no work line yet')
    }

    return node
  })
}

/** The header's toggle: a native button carrying aria-expanded. */
export const headerButton = (header: HTMLElement) => header.querySelector<HTMLButtonElement>('button[aria-expanded]')!

/** The folded line's label, without the trailing timestamp. */
export const headerLabel = (header: HTMLElement) => headerButton(header).textContent?.trim() ?? ''
