import { beforeEach, describe, expect, it } from 'vitest'

import { $unreadFinishedSessionIds, setSessionUnread } from './session'

describe('setSessionUnread', () => {
  beforeEach(() => $unreadFinishedSessionIds.set([]))

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
})
