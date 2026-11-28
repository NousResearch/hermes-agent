import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import {
  deleteWithThoughtRetirement,
  registerThoughtCapture,
  ThoughtCaptureStore,
  type ThoughtOwner
} from './thought-capture'

function bridge(directory: string) {
  const handlers = new Map<string, (event: { sender: { id: number } }, payload?: unknown) => any>()
  let owner: ThoughtOwner = { connectionId: 'remote-a', profile: 'work' }
  const store = new ThoughtCaptureStore(directory)

  const controller = registerThoughtCapture(
    {
      handle: (name, fn) => {
        handlers.set(name, fn)
      }
    },
    store,
    () => owner,
    id => id === 7
  )

  return {
    store,
    call: (name: string, payload?: unknown, sender = 7) =>
      handlers.get(`hermes:thoughts:${name}`)!({ sender: { id: sender } }, payload),
    switchTo: (next: ThoughtOwner) => {
      owner = next
      controller.invalidate()
    }
  }
}

describe('local thought capture through IPC', () => {
  it('persists exact drafts and idempotent saves across reload, owner switches and rename without gateway access', () => {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'thought-capture-'))

    try {
      const app = bridge(directory)
      const first = app.call('read')
      const draft = { id: 'thought-1', text: '  A thought — 日本語\nkeep spacing  ', handoffAttempted: true }
      app.call('draft', { token: first.token, draft })
      expect(bridge(directory).call('read').draft).toEqual(draft)
      const saved = app.call('save', { token: first.token, draft })
      expect(saved.thoughts.map((item: { text: string }) => item.text)).toEqual([draft.text])
      expect(app.call('save', { token: first.token, draft }).thoughts).toEqual(saved.thoughts)
      expect(() => app.call('save', { token: first.token, draft: { ...draft, text: 'different' } })).toThrow()
      app.switchTo({ connectionId: 'remote-b', profile: 'work' })
      expect(app.call('read').thoughts).toEqual([])
      expect(() => app.call('save', { token: first.token, draft })).toThrow()
      app.switchTo({ connectionId: 'remote-a', profile: 'work' })
      expect(() => app.call('save', { token: first.token, draft })).toThrow()
      expect(app.call('read').thoughts).toEqual(saved.thoughts)
      app.store.renameProfile('remote-a', 'work', 'renamed')
      app.switchTo({ connectionId: 'remote-a', profile: 'renamed' })
      expect(app.call('read').thoughts).toEqual(saved.thoughts)
      expect(() => app.call('read', undefined, 99)).toThrow()
      const reopened = app.call('read')
      app.call('draft', { token: reopened.token, draft: { id: 'large', text: 'another fragment' } })
      const scope = path.join(directory, readdirSync(directory)[0])
      const archive = path.join(scope, 'saved.json')
      const before = readFileSync(archive, 'utf8')
      app.call('draft', { token: reopened.token, draft: { id: 'large', text: 'edited fragment' } })
      expect(readFileSync(archive, 'utf8')).toBe(before)
      const files = readdirSync(directory)
      expect(files.every(name => !name.includes('remote') && !name.includes('work'))).toBe(true)
    } finally {
      rmSync(directory, { recursive: true, force: true })
    }
  })

  it('does not acknowledge failed writes or replace unreadable data with an empty inbox', () => {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'thought-failure-'))

    try {
      const blocked = path.join(directory, 'not-a-directory')
      writeFileSync(blocked, 'existing data')
      const app = bridge(blocked)
      const initial = app.call('read')
      expect(() => app.call('save', { token: initial.token, draft: { id: 'one', text: 'keep me' } })).toThrow()
      expect(readFileSync(blocked, 'utf8')).toBe('existing data')
      const valid = bridge(path.join(directory, 'valid'))
      const state = valid.call('read')
      valid.call('draft', { token: state.token, draft: { id: 'one', text: 'still here' } })
      const file = path.join(directory, 'valid', readdirSync(path.join(directory, 'valid'))[0], 'draft.json')
      writeFileSync(file, '{broken')
      expect(() => valid.call('read')).toThrow()
      expect(() => valid.call('save', { token: state.token, draft: { id: 'two', text: 'new' } })).toThrow()
      expect(readFileSync(file, 'utf8')).toBe('{broken')
    } finally {
      rmSync(directory, { recursive: true, force: true })
    }
  })
})

it('isolates a reused profile name, restores definite refusal, and retains uncertain deletions for recovery', async () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'thought-delete-'))

  try {
    const store = new ThoughtCaptureStore(directory)
    const owner = { connectionId: 'local', profile: 'work' }
    const draft = { id: 'one', text: 'retired profile thought' }
    store.save(owner, draft)
    const invalidate = vi.fn()
    const refuse = vi.fn(async () => ({ ok: false }))
    await deleteWithThoughtRetirement(store, owner, invalidate, refuse)
    expect(store.read(owner).thoughts[0].text).toBe(draft.text)
    await expect(
      deleteWithThoughtRetirement(store, owner, invalidate, async () => {
        throw new Error('timeout')
      })
    ).rejects.toThrow('preserved in')
    expect(store.read(owner).thoughts).toEqual([])
    expect(readdirSync(directory).some(name => name.includes('.retired-'))).toBe(true)
    store.save(owner, { id: 'new', text: 'new profile thought' })
    await deleteWithThoughtRetirement(store, owner, invalidate, async () => ({ ok: true }))
    expect(store.read(owner).thoughts).toEqual([])
    expect(invalidate).toHaveBeenCalledWith(owner)
    store.save(owner, { id: 'overlap', text: 'retain on refusal' })
    let refusePending!: (value: { ok: boolean }) => void

    const firstDelete = deleteWithThoughtRetirement(
      store,
      owner,
      invalidate,
      () =>
        new Promise(resolve => {
          refusePending = resolve
        })
    )

    const overlappingDelete = vi.fn(async () => ({ ok: true }))
    await expect(deleteWithThoughtRetirement(store, owner, invalidate, overlappingDelete)).rejects.toThrow(
      'already in progress'
    )
    expect(overlappingDelete).not.toHaveBeenCalled()
    refusePending({ ok: false })
    await firstDelete
    expect(store.read(owner).thoughts[0].text).toBe('retain on refusal')
    const remove = vi.fn(async () => ({ ok: true }))
    vi.spyOn(store, 'retireProfile').mockImplementationOnce(() => {
      throw new Error('disk failure')
    })
    await expect(deleteWithThoughtRetirement(store, owner, invalidate, remove)).rejects.toThrow('disk failure')
    expect(remove).not.toHaveBeenCalled()
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

it('refuses a retired owner until a new route activation, including a delete with no prior capture', async () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'thought-route-'))

  try {
    const app = bridge(directory)
    const owner = { connectionId: 'remote-a', profile: 'work' }
    await deleteWithThoughtRetirement(
      app.store,
      owner,
      () => {},
      async () => ({ ok: true })
    )
    expect(() => app.call('read')).toThrow('deleted')
    app.store.activateProfile(owner)
    expect(app.call('read').thoughts).toEqual([])
    await deleteWithThoughtRetirement(
      app.store,
      owner,
      () => {},
      async () => ({ ok: false })
    )
    expect(app.call('read').thoughts).toEqual([])
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
