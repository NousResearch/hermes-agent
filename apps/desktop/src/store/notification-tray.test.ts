import { atom } from 'nanostores'
import { beforeEach, expect, it, vi } from 'vitest'

import { $traySeen, createNotificationTray, type TrayItemState } from './notification-tray'

beforeEach(() => {
  window.localStorage.clear()
  $traySeen.set([])
})

const item = (id: string, state: null | TrayItemState) => ({ id, $state: atom<null | TrayItemState>(state) })

it('lists items in table order and leaves out the ones with nothing to show', () => {
  const update = item('update', null)
  const splash = item('splash', { news: null })
  const signUp = item('sign-up', { news: 'v1' })
  const tray = createNotificationTray([update, splash, signUp])

  expect(tray.$listed.get().map(entry => entry.id)).toEqual(['splash', 'sign-up'])

  update.$state.set({ news: '1.2.0' })
  expect(tray.$listed.get().map(entry => entry.id)).toEqual(['update', 'splash', 'sign-up'])
})

it('lights the bell for unseen news, clears it when seen, and lights it again for the next news', () => {
  const update = item('update', { news: '1.2.0' })
  const quiet = item('splash', { news: null })
  const tray = createNotificationTray([update, quiet])

  expect([...tray.$unseen.get()]).toEqual(['update'])

  tray.markAllSeen()
  expect(tray.$unseen.get().size).toBe(0)

  update.$state.set({ news: '1.3.0' })
  expect([...tray.$unseen.get()]).toEqual(['update'])
})

it('keeps what was seen across a reload of the page', async () => {
  createNotificationTray([item('update', { news: '1.2.0' })]).markAllSeen()

  vi.resetModules()
  const reloaded = await import('./notification-tray')
  const tray = reloaded.createNotificationTray([item('update', { news: '1.2.0' })])
  expect(tray.$unseen.get().size).toBe(0)
})

it('refuses two items with one id, whose seen news would collide', () => {
  expect(() => createNotificationTray([item('update', null), item('update', null)])).toThrow(/duplicate/)
})
