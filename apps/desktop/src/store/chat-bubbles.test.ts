// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $chatBubbles,
  CHAT_BUBBLE_DEFAULTS,
  CHAT_BUBBLE_ROOT_ATTR,
  CHAT_BUBBLE_TONE_ATTR,
  chatBubbleVars,
  normalizeChatBubbles,
  resetChatBubbles,
  sameChatBubbles,
  setChatBubbles
} from './chat-bubbles'

const KEY = 'hermes.desktop.chat-bubbles.v1'
const root = () => document.documentElement

describe('normalizeChatBubbles', () => {
  it('falls back to the document layout rather than trusting storage', () => {
    expect(normalizeChatBubbles(null)).toEqual(CHAT_BUBBLE_DEFAULTS)
    expect(normalizeChatBubbles('nonsense')).toEqual(CHAT_BUBBLE_DEFAULTS)
    expect(normalizeChatBubbles({ corners: 'squishy', layout: 'side-by-side', tone: 'neon' })).toEqual(CHAT_BUBBLE_DEFAULTS)
  })

  it('keeps every valid field of a partially valid state', () => {
    expect(normalizeChatBubbles({ corners: 'pill', layout: 'bubbles', tone: 'accent' })).toEqual({
      corners: 'pill',
      layout: 'bubbles',
      tone: 'accent'
    })

    expect(normalizeChatBubbles({ layout: 'bubbles', tone: 'neutral' })).toEqual({
      ...CHAT_BUBBLE_DEFAULTS,
      layout: 'bubbles',
      tone: 'neutral'
    })
  })
})

describe('chatBubbleVars', () => {
  it('paints the corner set the picker offers, tail included', () => {
    expect(chatBubbleVars({ ...CHAT_BUBBLE_DEFAULTS, corners: 'soft' })).toEqual({
      '--chat-bubble-radius': '0.625rem',
      '--chat-bubble-tail': '0.25rem'
    })

    expect(chatBubbleVars({ ...CHAT_BUBBLE_DEFAULTS, corners: 'pill' })).toEqual({
      '--chat-bubble-radius': '1.5rem',
      '--chat-bubble-tail': '0.375rem'
    })
  })
})

describe('sameChatBubbles', () => {
  it('compares field-wise, so an equal state is recognized as a no-op', () => {
    const state = { ...CHAT_BUBBLE_DEFAULTS, layout: 'bubbles' as const }

    expect(sameChatBubbles(state, { ...state })).toBe(true)
    expect(sameChatBubbles(state, { ...state, corners: 'pill' })).toBe(false)
    expect(sameChatBubbles(state, { ...state, tone: 'accent' })).toBe(false)
    expect(sameChatBubbles(state, { ...state, layout: 'document' })).toBe(false)
  })
})

describe('the chat layout on the root element', () => {
  beforeEach(resetChatBubbles)

  it('marks the root for bubbles and clears it for the document layout', () => {
    setChatBubbles({ layout: 'bubbles', corners: 'soft', tone: 'neutral' })

    expect(root().getAttribute(CHAT_BUBBLE_ROOT_ATTR)).toBe('bubbles')
    expect(root().getAttribute(CHAT_BUBBLE_TONE_ATTR)).toBe('neutral')
    expect(root().style.getPropertyValue('--chat-bubble-radius')).toBe('0.625rem')
    expect(root().style.getPropertyValue('--chat-bubble-tail')).toBe('0.25rem')

    setChatBubbles({ layout: 'document' })

    expect(root().hasAttribute(CHAT_BUBBLE_ROOT_ATTR)).toBe(false)
    expect(root().hasAttribute(CHAT_BUBBLE_TONE_ATTR)).toBe(false)
  })

  it('patches one concern without disturbing the others', () => {
    setChatBubbles({ layout: 'bubbles', corners: 'round', tone: 'accent' })
    setChatBubbles({ corners: 'pill' })

    expect($chatBubbles.get()).toEqual({ corners: 'pill', layout: 'bubbles', tone: 'accent' })
    expect(root().style.getPropertyValue('--chat-bubble-radius')).toBe('1.5rem')
  })

  it('never reaches a transcript node — the shape comes from the stylesheet', () => {
    const transcript = document.createElement('div')
    const bubble = document.createElement('button')

    transcript.setAttribute('data-slot', 'aui_user-message-root')
    bubble.className = 'composer-human-message'
    transcript.appendChild(bubble)
    document.body.appendChild(transcript)

    setChatBubbles({ layout: 'bubbles' })

    // State lives on <html>; no message node carries layout state we would then
    // have to keep in sync (and re-render the virtualized thread to update).
    expect(transcript.getAttribute('style')).toBeNull()
    expect(bubble.getAttribute('style')).toBeNull()
    expect(bubble.className).toBe('composer-human-message')

    document.body.removeChild(transcript)
  })

  it('persists under the shared desktop key', () => {
    setChatBubbles({ layout: 'bubbles', corners: 'pill', tone: 'neutral' })

    expect(JSON.parse(window.localStorage.getItem(KEY) ?? 'null')).toEqual({
      corners: 'pill',
      layout: 'bubbles',
      tone: 'neutral'
    })

    // The document layout is a value, not an absence: the row must be able to
    // read the choice back after a restart.
    setChatBubbles({ layout: 'document' })

    expect(JSON.parse(window.localStorage.getItem(KEY) ?? 'null')).toEqual({
      corners: 'pill',
      layout: 'document',
      tone: 'neutral'
    })
  })
})

/**
 * localStorage is shared by every renderer window of this profile, so this
 * preference is desktop-wide rather than per window. That makes cross-window
 * reconciliation part of the contract: the browser fires `storage` in the
 * windows that did NOT write, and this store has to repaint from it instead of
 * disagreeing until the next launch.
 */
describe('cross-window reconciliation', () => {
  beforeEach(resetChatBubbles)

  it('follows a layout another window wrote', () => {
    setChatBubbles({ layout: 'bubbles', corners: 'round', tone: 'theme' })

    const external = { corners: 'pill', layout: 'document', tone: 'accent' }
    window.localStorage.setItem(KEY, JSON.stringify(external))
    window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: JSON.stringify(external) }))

    expect($chatBubbles.get()).toEqual(external)
    expect(root().hasAttribute(CHAT_BUBBLE_ROOT_ATTR)).toBe(false)
  })

  it('normalizes what another window left behind', () => {
    window.localStorage.setItem(KEY, JSON.stringify({ corners: 'squishy', layout: 'bubbles', tone: 'neon' }))
    window.dispatchEvent(new StorageEvent('storage', { key: KEY }))

    expect($chatBubbles.get()).toEqual({ ...CHAT_BUBBLE_DEFAULTS, layout: 'bubbles' })
  })

  it('does not write back a no-op event — two windows must not ping-pong', () => {
    setChatBubbles({ layout: 'bubbles', corners: 'pill', tone: 'accent' })

    const write = vi.spyOn(window.localStorage, 'setItem')
    const current = window.localStorage.getItem(KEY)

    try {
      window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: current }))
      expect(write).not.toHaveBeenCalled()
    } finally {
      write.mockRestore()
    }
  })

  it('ignores keys that are not ours', () => {
    setChatBubbles({ layout: 'bubbles', corners: 'round', tone: 'theme' })

    window.localStorage.setItem('some.other.key', '{"layout":"document"}')
    window.dispatchEvent(new StorageEvent('storage', { key: 'some.other.key', newValue: '{"layout":"document"}' }))

    expect($chatBubbles.get().layout).toBe('bubbles')
  })
})
