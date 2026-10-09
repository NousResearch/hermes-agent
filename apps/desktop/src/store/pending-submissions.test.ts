import { beforeEach, expect, it } from 'vitest'

import { applyRuntimeInfo } from '@/app/session/hooks/use-session-actions/utils'
import { isSteerableEntry } from '@/store/composer-queue'

import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts } from './composer-queue'
import { reconcilePendingSubmissions, trackPendingSubmission } from './pending-submissions'

beforeEach(() => {
  window.localStorage.clear()
  $queuedPromptsBySession.set({})
})

it('projects resumed runtime pending receipts under durable identity without making them sendable', () => {
  const info = {
    stored_session_id: 'durable',
    pending_submissions: [
      { admission_id: 'remote', status: 'queued', user: 'queued exact Ω' },
      { admission_id: 'uncertain', status: 'unknown', user: 'unknown exact Ω' }
    ]
  }

  applyRuntimeInfo(info)
  applyRuntimeInfo(info)
  expect(getQueuedPrompts('durable').map(({ id, text, serverStatus }) => ({ id, text, serverStatus })))
    .toEqual(info.pending_submissions.map(({ admission_id, user, status }) => ({ id: admission_id, text: user, serverStatus: status })))
  expect(getQueuedPrompts('durable').every(entry => !isSteerableEntry(entry))).toBe(true)
  expect(getQueuedPrompts('runtime')).toEqual([])
})

it('reconciles server queue by identity without replaying or duplicating local entries', () => {
  enqueueQueuedPrompt('chat', { id: 'one', text: 'same', attachments: [] })
  enqueueQueuedPrompt('chat', { id: 'local', text: 'same', attachments: [] })
  const snapshot = [{ admission_id: 'one', status: 'queued', user: 'same' }]
  reconcilePendingSubmissions('chat', snapshot)
  reconcilePendingSubmissions('chat', snapshot)
  expect(getQueuedPrompts('chat').map(entry => entry.id)).toEqual(['one', 'local'])
  expect(getQueuedPrompts('chat')[0]?.serverStatus).toBe('queued')
  reconcilePendingSubmissions('chat', [{ ...snapshot[0], status: 'started' }])
  expect(getQueuedPrompts('chat').map(entry => entry.id)).toEqual(['local'])
  reconcilePendingSubmissions('chat', [{ admission_id: 'unknown', status: 'unknown', user: 'interrupted' }])
  expect(getQueuedPrompts('chat').find(entry => entry.id === 'unknown')?.serverStatus).toBe('unknown')
  reconcilePendingSubmissions('chat', [])
  expect(getQueuedPrompts('chat').map(entry => entry.id)).toEqual(['local'])
})

it('recovers remote pending text into the queue and journal without a local submission', () => {
  const snapshot = [
    { admission_id: 'remote-queued', status: 'queued', user: 'queued elsewhere' },
    { admission_id: 'remote-unknown', status: 'unknown', user: 'interrupted elsewhere' }
  ]

  reconcilePendingSubmissions('chat', snapshot)
  reconcilePendingSubmissions('chat', snapshot)
  expect(getQueuedPrompts('chat').map(({ id, text, serverStatus }) => ({ id, text, serverStatus })))
    .toEqual(snapshot.map(({ admission_id, user, status }) => ({ id: admission_id, text: user, serverStatus: status })))
  const journal = JSON.parse(window.localStorage.getItem('hermes.desktop.pendingSubmissions.v1')!)

  for (const receipt of snapshot) {
    expect(journal.chat[receipt.admission_id].text).toBe(receipt.user)
  }
})

it('maps optimistic input identity to the admission identity before local drain can replay it', () => {
  enqueueQueuedPrompt('mapped', { id: 'input-id', text: 'same', attachments: [] })
  const receipt = { admission_id: 'admission-id', input_id: 'input-id', status: 'queued', user: 'same' }
  reconcilePendingSubmissions('mapped', [receipt])
  expect(getQueuedPrompts('mapped')).toMatchObject([{ id: 'admission-id', serverStatus: 'queued', text: 'same' }])
  expect(getQueuedPrompts('mapped')).toHaveLength(1)
  reconcilePendingSubmissions('mapped', [{ ...receipt, status: 'started' }])
  expect(getQueuedPrompts('mapped')).toEqual([])
})

it('persists identified direct submissions independently of the automatic local queue', () => {
  trackPendingSubmission('chat', { id: 'direct', text: 'hello' })
  const stored = JSON.parse(window.localStorage.getItem('hermes.desktop.pendingSubmissions.v1')!)
  expect(stored.chat.direct).toMatchObject({ id: 'direct', text: 'hello' })
  expect(getQueuedPrompts('chat')).toEqual([])
})
