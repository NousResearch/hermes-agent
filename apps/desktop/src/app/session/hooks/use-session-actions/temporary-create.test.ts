import { beforeEach, expect, test, vi } from 'vitest'

import { $temporaryChats, $temporaryDraft, isTemporarySession } from '@/store/temporary-chat'

import { adoptTemporarySession, temporaryCreateParams } from './temporary-create'

beforeEach(() => {
  $temporaryDraft.set(false)
  $temporaryChats.set({})
})

test('only a temporary draft asks session.create for a temporary session', () => {
  expect(temporaryCreateParams()).toEqual({})
  $temporaryDraft.set(true)
  expect(temporaryCreateParams()).toEqual({ temporary: true })
})

test('a temporary create is adopted by id instead of seeding a sidebar row', () => {
  expect(adoptTemporarySession({ source: 'desktop' }, 'rt-1', 'stored-1')).toBe(false)
  expect(isTemporarySession('stored-1')).toBe(false)

  expect(adoptTemporarySession({ temporary: true }, 'rt-2', 'stored-2')).toBe(true)
  expect(isTemporarySession('stored-2')).toBe(true)
  expect(isTemporarySession('rt-2')).toBe(true)
})

test('seeds a sidebar row for a saved chat only', async () => {
  const { seedCreatedSession } = await import('./temporary-create')
  const seed = vi.fn()
  seedCreatedSession({ source: 'desktop' }, 'rt-3', 'stored-3', seed)
  expect(seed).toHaveBeenCalledTimes(1)
  seedCreatedSession({ temporary: true }, 'rt-4', 'stored-4', seed)
  expect(seed).toHaveBeenCalledTimes(1)
})
