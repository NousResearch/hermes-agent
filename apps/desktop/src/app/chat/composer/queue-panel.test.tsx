import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { QueuedPromptEntry } from '@/store/composer-queue'

import { QueuePanel } from './queue-panel'

const entry = (id: string, text: string): QueuedPromptEntry => ({
  attachments: [],
  id,
  queuedAt: 1,
  text
})

function renderQueuePanel(
  props: Partial<ComponentProps<typeof QueuePanel>> = {}
) {
  const onDelete = vi.fn()
  const onEdit = vi.fn()
  const onSendAll = vi.fn()
  const onSendNow = vi.fn()

  render(
    <I18nProvider configClient={null}>
      <QueuePanel
        busy={false}
        editingId={null}
        entries={[entry('q1', 'first'), entry('q2', 'second')]}
        onDelete={onDelete}
        onEdit={onEdit}
        onSendAll={onSendAll}
        onSendNow={onSendNow}
        {...props}
      />
    </I18nProvider>
  )

  return { onDelete, onEdit, onSendAll, onSendNow }
}

describe('QueuePanel send all action', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders a header action for sending all queued messages', () => {
    renderQueuePanel()

    expect(screen.getByRole('button', { name: 'Send all' })).toBeTruthy()
  })

  it('calls onSendAll from the header action', () => {
    const { onSendAll } = renderQueuePanel()

    fireEvent.click(screen.getByRole('button', { name: 'Send all' }))

    expect(onSendAll).toHaveBeenCalledTimes(1)
  })

  it('uses the next label while a turn is busy', () => {
    renderQueuePanel({ busy: true })

    expect(screen.getByRole('button', { name: 'Send all next' })).toBeTruthy()
  })

  it('disables send all while a queued message is being edited', () => {
    renderQueuePanel({ editingId: 'q1' })

    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Send all' }).disabled).toBe(true)
  })
})
