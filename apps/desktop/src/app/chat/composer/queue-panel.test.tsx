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

function renderPanel(entries: QueuedPromptEntry[], requestedDaybreak?: boolean) {
  return render(
    <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
      <QueuePanel
        busy={requestedDaybreak !== undefined}
        editingId={null}
        entries={entries}
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onResume={vi.fn()}
        onSendNow={vi.fn()}
        onSteerNow={vi.fn()}
        parked={false}
        requestedDaybreak={requestedDaybreak}
        runningDaybreak={true}
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


it('offers redirect only when the current selection matches the running program', async () => {
  let view = renderPanel([entry('queued', 'review')], false)
  fireEvent.click(await view.findByRole('button', { name: /queued/i }))
  expect(screen.queryByRole('button', { name: /Steer —/ })).toBeNull()
  cleanup()
  view = renderPanel([entry('queued', 'review')], true)
  fireEvent.click(await view.findByRole('button', { name: /queued/i }))
  expect(screen.getByRole('button', { name: /Steer —/ })).toBeTruthy()
})
