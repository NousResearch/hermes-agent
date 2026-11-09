import { afterEach, expect, it, vi } from 'vitest'

import { makeSessionInfo } from '../test/session-info'

afterEach(() => window.localStorage.removeItem('hermes.desktop.sessionUnreadMutes'))

it('restores a saved mute before projecting backend unread on a cold load', async () => {
  window.localStorage.setItem('hermes.desktop.sessionUnreadMutes', JSON.stringify({ default: ['root'] }))
  vi.resetModules()
  const { setSessions } = await import('./session')
  const { $sessionDotStateById, $unreadSessionCount } = await import('./session-dot-state')
  setSessions([makeSessionInfo({ id: 'tip', _lineage_root_id: 'root', unread: true })])
  expect($sessionDotStateById.get().tip).not.toBe('unread')
  expect($unreadSessionCount.get()).toBe(0)
})
