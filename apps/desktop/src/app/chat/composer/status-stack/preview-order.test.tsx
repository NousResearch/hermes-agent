import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { $backgroundStatusBySession } from '@/store/composer-status'
import { $previewStatusBySession } from '@/store/preview-status'
import { $todosBySession } from '@/store/todos'

import { QueuePanel } from '../queue-panel'

import { ComposerStatusStack } from './index'

const noop = () => {}

const queue = (
  <QueuePanel
    busy
    editingId={null}
    entries={[{ attachments: [], id: 'queued', queuedAt: 1, text: 'Queued request' }]}
    onDelete={noop}
    onEdit={noop}
    onResume={noop}
    onSendNow={noop}
    parked={false}
  />
)

const stack = (queued: boolean) => (
  <MemoryRouter>
    <ComposerStatusStack queue={queued ? queue : null} sessionId="owner" />
  </MemoryRouter>
)

function expectBefore(before: HTMLElement, after: HTMLElement) {
  expect(before.compareDocumentPosition(after) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
}

afterEach(() => {
  cleanup()
  $backgroundStatusBySession.set({})
  $previewStatusBySession.set({})
  $todosBySession.set({})
})

it.each([false, true])('keeps artifact links below the queue with background work: %s', background => {
  $todosBySession.set({ owner: [{ id: 'todo', content: 'Task item', status: 'in_progress' }] })
  $previewStatusBySession.set({
    owner: [
      { cwd: '/tmp', id: 'file', label: 'index.html', target: '/tmp/index.html' },
      { cwd: '/tmp', id: 'url', label: 'Live preview', target: 'http://localhost:5174' }
    ]
  })

  if (background) {
    $backgroundStatusBySession.set({
      owner: [{ id: 'process', type: 'background', state: 'running', title: 'Preview server' }]
    })
  }

  const view = render(stack(true))
  const file = screen.getByText('index.html')
  const header = screen.getByRole('button', { name: /1 Queued/ })
  expectBefore(screen.getByText('Task item'), file)
  expectBefore(header, file)
  expect(screen.queryAllByText('Live preview')).toHaveLength(background ? 1 : 0)

  fireEvent.click(header)
  expectBefore(screen.getByText('Queued request'), file)

  if (background) {
    expectBefore(file, screen.getByText('Live preview'))
    act(() => $backgroundStatusBySession.set({}))
    expect(screen.queryByText('Live preview')).toBeNull()
    expectBefore(screen.getByText('Queued request'), file)
  }

  view.rerender(stack(false))
  expectBefore(screen.getByText('Task item'), file)
  view.rerender(stack(true))
  expectBefore(screen.getByRole('button', { name: /1 Queued/ }), file)
})
