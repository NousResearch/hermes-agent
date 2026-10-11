import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import type { QueuedPromptEntry } from '@/store/composer-queue'

import { QueuePanel } from './queue-panel'

const entry = (id: string, text: string): QueuedPromptEntry => ({
  attachments: [],
  id,
  queuedAt: Date.now(),
  text
})

function renderPanel(
  entries: QueuedPromptEntry[],
  overrides: { editingId?: null | string; onMergeAll?: () => void; parked?: boolean } = {}
) {
  return render(
    <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
      <QueuePanel
        busy={false}
        editingId={overrides.editingId ?? null}
        entries={entries}
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onMergeAll={overrides.onMergeAll}
        onResume={vi.fn()}
        onSendNow={vi.fn()}
        parked={overrides.parked ?? false}
      />
    </I18nProvider>
  )
}

afterEach(cleanup)

// The panel's StatusSection starts collapsed; tests read the entries, so open it.
async function renderExpandedPanel(entries: QueuedPromptEntry[]) {
  const view = renderPanel(entries)

  fireEvent.click(await view.findByRole('button', { name: /queued/i }))

  return view
}

// #45664: a long queued prompt was only ever readable as one truncated line
// (or by loading it back into the composer). Long and multiline previews now
// carry an expand/collapse toggle that reveals the full text in place.
describe('QueuePanel expandable previews', () => {
  it('offers expansion for long and multiline entries but not short ones', async () => {
    await renderExpandedPanel([
      entry('short', 'Quick note'),
      entry('long', 'A'.repeat(200)),
      entry('multiline', 'First line\nSecond line')
    ])

    await screen.findByText('Quick note')

    // Exactly the two long/multiline entries get a toggle, not the short one.
    expect(screen.getAllByRole('button', { name: /expand/i })).toHaveLength(2)
  })

  it('expands an entry to its full text and collapses it back', async () => {
    const text = `${'A'.repeat(200)}\nlast line`
    await renderExpandedPanel([entry('long', text)])

    // The DOM keeps the newline in textContent; findByText normalizes
    // whitespace, so locate the <p> by exact textContent instead.
    expect(
      await screen.findByText((_, element) => element?.tagName === 'P' && element.textContent === text)
    ).toBeTruthy()

    const toggle = screen.getByRole('button', { name: /expand/i })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(toggle)

    const collapse = screen.getByRole('button', { name: /collapse/i })
    expect(collapse.getAttribute('aria-expanded')).toBe('true')

    fireEvent.click(collapse)
    expect(screen.getByRole('button', { name: /expand/i }).getAttribute('aria-expanded')).toBe('false')
  })
})

// #41247: a queue of several short follow-ups is usually one instruction. The
// header's merge button folds them into a single entry in place.
describe('QueuePanel merge affordance', () => {
  it('offers the merge button for two or more plain queued turns and fires the host hook', () => {
    const onMergeAll = vi.fn()

    renderPanel([entry('a', 'one'), entry('b', 'two')], { onMergeAll })

    fireEvent.click(screen.getByRole('button', { name: /merge all queued turns/i }))

    expect(onMergeAll).toHaveBeenCalledTimes(1)
  })

  it('hides the merge button for a single entry and when the host has no merge path', () => {
    renderPanel([entry('a', 'one'), entry('b', 'two')])

    expect(screen.queryByRole('button', { name: /merge all queued turns/i })).toBeNull()

    cleanup()
    renderPanel([entry('a', 'one')], { onMergeAll: vi.fn() })

    expect(screen.queryByRole('button', { name: /merge all queued turns/i })).toBeNull()
  })

  it('hides the merge button while an entry sits in the composer for editing', () => {
    renderPanel([entry('a', 'one'), entry('b', 'two')], { editingId: 'b', onMergeAll: vi.fn() })

    expect(screen.queryByRole('button', { name: /merge all queued turns/i })).toBeNull()
  })

  it('hides the merge button when an entry cannot be folded losslessly', () => {
    renderPanel([entry('a', 'one'), entry('chip', 'look at @terminal:`zsh:23-58`')], { onMergeAll: vi.fn() })

    expect(screen.queryByRole('button', { name: /merge all queued turns/i })).toBeNull()

    cleanup()
    renderPanel([entry('a', 'one'), { ...entry('note', 'setup'), displayKind: 'hidden' }], { onMergeAll: vi.fn() })

    expect(screen.queryByRole('button', { name: /merge all queued turns/i })).toBeNull()
  })

  it('keeps the resume affordance alongside the merge button on a parked queue', () => {
    renderPanel([entry('a', 'one'), entry('b', 'two')], { onMergeAll: vi.fn(), parked: true })

    expect(screen.getByRole('button', { name: /merge all queued turns/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /resume/i })).toBeTruthy()
  })
})
