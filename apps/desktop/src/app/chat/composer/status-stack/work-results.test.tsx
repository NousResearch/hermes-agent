import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { ComposerStatusStack } from './index'

// A finished worker's report is a model answer with tables, bold and code in
// it — and it used to be painted right above the composer (first as
// `whitespace-pre-wrap` text, later as markdown). Either way the user was
// reading what Hermes answered. The lane now shows the animated huddle of the
// agents talking instead; the report still exists in the store (and is spoken
// by the Live agent — see agent-huddle-loop.test.tsx).

vi.mock('@/lib/use-enter-animation', () => ({ useEnterAnimation: () => undefined }))

vi.stubGlobal(
  'ResizeObserver',
  class {
    disconnect() {}
    observe() {}
    unobserve() {}
  }
)

afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
})

const REPORT = [
  'Sprawdziłem historię — o pamięci gadałem Ci tylko przelotnie:',
  '',
  '| Warstwa | Gdzie siedzi | Stan u Ciebie |',
  '|---|---|---|',
  '| Pamięć trwała (`user` + `memory`) | `hermes-home/memories/` | **pusta** — zero faktów |',
  '',
  '- **Vault** nie istnieje, nie ustaliliśmy ścieżki',
  '- Skille (procedury) w `hermes-home/skills/`'
].join('\n')

function renderStack() {
  return render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="owner" />
    </MemoryRouter>
  )
}

it('shows the agent huddle instead of the finished worker report', () => {
  act(() => {
    upsertSubagent('owner', {
      files_written: ['C:/Users/ostry/hermes-vault/CORE_MEMORY.md'],
      goal: 'Zbadaj system pamięci',
      status: 'completed',
      subagent_id: 'child-1',
      summary: REPORT
    })
  })

  const { container } = renderStack()
  const text = container.textContent ?? ''

  expect(container.querySelector('[data-testid="agent-huddle"]')?.getAttribute('data-state')).toBe('done')
  expect(text).toContain('Narada zakończona')
  // Neither the markdown source, nor its rendered prose, nor the file list.
  expect(text).not.toContain('**pusta**')
  expect(text).not.toContain('|---|')
  expect(text).not.toContain('hermes-home/memories/')
  expect(text).not.toContain('Pamięć trwała')
  expect(text).not.toContain('CORE_MEMORY.md')
  expect(container.querySelector('pre')).toBeNull()
  expect(container.querySelector('table')).toBeNull()
  expect(container.querySelector('[data-slot="subagent-transcript"]')).toBeNull()
})

it('stands the agents up while the work is still running', () => {
  act(() => {
    upsertSubagent('owner', { goal: 'Zbadaj system pamięci', status: 'running', subagent_id: 'child-3' })
    upsertSubagent('owner', { subagent_id: 'child-3', text: 'Czytam pliki' }, false, 'subagent.progress')
  })

  const { container } = renderStack()
  const huddle = container.querySelector('[data-testid="agent-huddle"]')

  expect(huddle?.getAttribute('data-state')).toBe('talking')
  expect(container.querySelector('[data-testid="agent-huddle"]')).toBeTruthy()
})

it('has no lane at all when no worker is on the roster', () => {
  const { container } = renderStack()

  expect(container.querySelector('[data-slot="composer-status-stack"]')).toBeNull()
  expect(screen.queryByTestId('agent-huddle')).toBeNull()
})
