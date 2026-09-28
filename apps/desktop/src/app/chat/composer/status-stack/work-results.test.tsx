import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { ComposerStatusStack } from './index'

// A finished worker's report is a model answer with tables, bold and code in
// it. It used to be painted as `whitespace-pre-wrap` text, so the panel above
// the composer showed the markdown source: literal `**bold**`, `| a | b |` and
// backticks. It has to render as markdown.

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

it('renders a finished worker report as rich markdown, not the markdown source', async () => {
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

  await waitFor(() => {
    expect(container.querySelector('table')).toBeTruthy()
  })

  const text = container.textContent ?? ''

  expect(text).not.toContain('**pusta**')
  expect(text).not.toContain('|---|')
  expect(text).not.toContain('`hermes-home/memories/`')
  expect(container.querySelector('th')?.textContent).toContain('Warstwa')
  expect(container.querySelectorAll('tbody tr').length).toBe(1)
  expect(container.querySelector('strong, [data-streamdown="strong"], .font-semibold')).toBeTruthy()
  expect(container.querySelector('li')?.textContent).toContain('Vault')
  expect(container.textContent).toContain('Pamięć trwała (user + memory)')
  expect(container.textContent).toContain('pusta — zero faktów')
})

it('keeps the no-report fallback as plain copy', () => {
  act(() => {
    upsertSubagent('owner', { goal: 'Zbadaj system pamięci', status: 'failed', subagent_id: 'child-2' })
  })

  renderStack()

  expect(screen.getByText(/Backend nie dostarczył raportu wyniku/)).toBeTruthy()
})
