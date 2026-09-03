import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $hudRoom, $hudRoomFeed, enterHudRoom, leaveHudRoom, postHudRoom, watchHudRoomFeed } from './hud'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initial = desktopWindow.hermesDesktop

const roomFeed = vi.fn()
const roomPost = vi.fn()
const watchRoom = vi.fn()
let feedListener: ((feed: unknown) => void) | null = null

const feed = (groupId: string) => ({
  groupId,
  entries: [{ id: '1', at: 1, author: 'You', kind: 'user' as const, text: 'hi' }],
  members: ['gary'],
  running: false,
  turn: null
})

beforeEach(() => {
  vi.clearAllMocks()
  roomFeed.mockImplementation(async (groupId: string) => feed(groupId))
  roomPost.mockResolvedValue({ ok: true })
  watchRoom.mockResolvedValue(undefined)
  feedListener = null
  desktopWindow.hermesDesktop = {
    hud: {
      roomFeed,
      roomPost,
      watchRoom,
      onRoomFeed: (cb: (feed: unknown) => void) => {
        feedListener = cb

        return () => {
          feedListener = null
        }
      }
    }
  } as unknown as Window['hermesDesktop']
  $hudRoom.set(null)
  $hudRoomFeed.set(null)
})

afterEach(() => {
  if (initial) {
    desktopWindow.hermesDesktop = initial
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('HUD room mode', () => {
  it('enters a room: watches it in the app window and loads the feed', async () => {
    expect(await enterHudRoom(' Design ')).toBe(true)
    expect($hudRoom.get()).toBe('Design')
    expect(watchRoom).toHaveBeenCalledWith('Design')
    expect($hudRoomFeed.get()?.entries).toHaveLength(1)
  })

  it('posts into the entered room and refuses outside one', async () => {
    expect(await postHudRoom('hello')).toBe(false)
    await enterHudRoom('Design')
    expect(await postHudRoom('  hello  ')).toBe(true)
    expect(roomPost).toHaveBeenCalledWith('Design', 'hello')
    expect(await postHudRoom('   ')).toBe(false)
  })

  it('leaves: clears state and stops watching', async () => {
    await enterHudRoom('Design')
    leaveHudRoom()
    expect($hudRoom.get()).toBeNull()
    expect($hudRoomFeed.get()).toBeNull()
    expect(watchRoom).toHaveBeenLastCalledWith(null)
  })

  it('takes live feed pushes only for the entered room', async () => {
    const off = watchHudRoomFeed()
    await enterHudRoom('Design')
    feedListener?.(feed('Other'))
    expect($hudRoomFeed.get()?.groupId).toBe('Design')
    feedListener?.({ ...feed('Design'), running: true, turn: 'gary' })
    expect($hudRoomFeed.get()?.turn).toBe('gary')
    off()
    expect(feedListener).toBeNull()
  })
})
