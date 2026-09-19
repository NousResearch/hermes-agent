import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { QueuePanel } from '@/app/chat/composer/queue-panel'
import { PreviewStatusRow } from '@/app/chat/composer/status-stack/preview-row'
import { StatusItemRow } from '@/app/chat/composer/status-stack/status-row'
import { Codicon } from '@/components/ui/codicon'
import { I18nProvider } from '@/i18n'

import { StatusRow } from './status-row'

vi.stubGlobal(
  'ResizeObserver',
  class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
)

afterEach(cleanup)

it('keeps dismiss and nested controls independent from row activation', () => {
  const activate = vi.fn()
  const dismiss = vi.fn()
  const action = vi.fn()

  const { container } = render(
    <div data-slot="composer-status-stack">
      <StatusRow
        dismiss={{ label: 'Dismiss task', onDismiss: dismiss }}
        leading={<Codicon name="comment" />}
        onActivate={activate}
        trailing={
          <button
            onClick={event => {
              event.stopPropagation()
              action()
            }}
          >
            Edit task
          </button>
        }
      >
        <span>Task title</span>
      </StatusRow>
    </div>
  )

  const row = container.querySelector<HTMLElement>('[data-slot="status-row"]')!
  const close = screen.getByRole('button', { name: 'Dismiss task' })
  fireEvent.keyDown(close, { key: 'Enter' })
  fireEvent.click(close)
  expect(dismiss).toHaveBeenCalledOnce()
  expect(activate).not.toHaveBeenCalled()
  fireEvent.keyDown(screen.getByRole('button', { name: 'Edit task' }), { key: ' ' })
  fireEvent.click(screen.getByRole('button', { name: 'Edit task' }))
  expect(action).toHaveBeenCalledOnce()
  expect(activate).not.toHaveBeenCalled()
  fireEvent.keyDown(row, { key: 'Enter' })
  fireEvent.keyDown(row, { key: ' ' })
  expect(activate).toHaveBeenCalledTimes(2)
})

it('uses the same leading dismiss control for queued messages, artifacts and background work', () => {
  const deleteQueued = vi.fn()
  const dismissArtifact = vi.fn()
  const stopBackground = vi.fn()

  const { container } = render(
    <I18nProvider configClient={null} initialLocale="en">
      <div data-slot="composer-status-stack">
        <QueuePanel
          busy={false}
          editingId={null}
          entries={[{ id: 'queued', text: 'Queued message', attachments: [], queuedAt: 0 }]}
          onDelete={deleteQueued}
          onEdit={vi.fn()}
          onResume={vi.fn()}
          onSendNow={vi.fn()}
          parked
        />
        <PreviewStatusRow
          item={{ id: 'artifact', label: 'report.html', target: '/tmp/report.html', cwd: '/tmp' }}
          onDismiss={dismissArtifact}
        />
        <StatusItemRow
          item={{ id: 'process', title: 'Background task', type: 'background', state: 'running' }}
          onStop={stopBackground}
        />
      </div>
    </I18nProvider>
  )

  fireEvent.click(screen.getByRole('button', { name: /1 Queued/ }))
  const buttons = [...container.querySelectorAll<HTMLButtonElement>('[data-slot="status-dismiss"]')]
  expect(buttons).toHaveLength(3)
  const reference = buttons[0]!

  for (const button of buttons) {
    expect(button.className).toBe(reference.className)
    expect(button.querySelector('.codicon-close')?.outerHTML).toBe(reference.querySelector('.codicon-close')?.outerHTML)
    expect(button.closest('[data-slot="status-row"]')?.firstElementChild?.contains(button)).toBe(true)
  }

  fireEvent.click(within(container).getByRole('button', { name: 'Delete' }))
  fireEvent.click(within(container).getByRole('button', { name: 'Dismiss' }))
  fireEvent.click(within(container).getByRole('button', { name: 'Stop' }))
  expect(deleteQueued).toHaveBeenCalledWith('queued')
  expect(dismissArtifact).toHaveBeenCalledWith('artifact')
  expect(stopBackground).toHaveBeenCalledWith('process')
})
