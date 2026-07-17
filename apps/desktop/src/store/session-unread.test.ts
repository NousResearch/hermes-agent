import { beforeEach, describe, expect, it } from 'vitest'

import {
  $unreadFinishedSessionIds,
  clearAllSessionUnread,
  getSessionCompletionToken,
  getSessionRenderedCompletion,
  sessionHasUnread,
  sessionLineageIds,
  setSessions,
  setSessionUnread
} from './session'

describe('setSessionUnread', () => {
  beforeEach(() => {
    clearAllSessionUnread()
    setSessions([])
  })

  it('adds and removes a session id without duplicating it', () => {
    setSessionUnread('s1', true)
    setSessionUnread('s1', true)
    setSessionUnread('s2', true)

    expect($unreadFinishedSessionIds.get()).toEqual(['s1', 's2'])

    setSessionUnread('s1', false)
    expect($unreadFinishedSessionIds.get()).toEqual(['s2'])
  })

  it('ignores missing ids and no-op clears', () => {
    setSessionUnread(null, true)
    setSessionUnread(undefined, true)
    setSessionUnread('', true)
    setSessionUnread('missing', false)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('keeps every observed compression alias bound to the lineage root', () => {
    setSessions([{ _lineage_root_id: 'root', id: 'tip-1' }] as never)
    setSessionUnread('tip-1', true)
    setSessions([{ _lineage_root_id: 'root', id: 'tip-2' }] as never)

    expect(sessionLineageIds('tip-2')).toEqual(['tip-2', 'root', 'tip-1'])
    expect($unreadFinishedSessionIds.get()).toEqual(['root'])
  })

  it('keeps identical raw session ids isolated by profile', () => {
    setSessionUnread('same', true, 'alpha')

    expect(sessionHasUnread('same', 'alpha')).toBe(true)
    expect(sessionHasUnread('same', 'beta')).toBe(false)

    setSessionUnread('same', true, 'beta')
    setSessionUnread('same', false, 'alpha')

    expect(sessionHasUnread('same', 'alpha')).toBe(false)
    expect(sessionHasUnread('same', 'beta')).toBe(true)
    expect($unreadFinishedSessionIds.get()).toEqual(['same'])
  })

  it('resolves the rendered completion by exact profile when raw ids collide', () => {
    setSessionUnread('same', true, 'alpha')
    setSessionUnread('same', true, 'beta')

    expect(getSessionRenderedCompletion('same', 'alpha')).toEqual({
      completion: getSessionCompletionToken('same', 'alpha'),
      profile: 'alpha'
    })
    expect(getSessionRenderedCompletion('same', 'beta')).toEqual({
      completion: getSessionCompletionToken('same', 'beta'),
      profile: 'beta'
    })
    expect(getSessionRenderedCompletion('same', 'gamma')).toBeNull()
  })
})
