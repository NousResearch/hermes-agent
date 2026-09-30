// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $chatTextSize, setChatTextSize } from '@/store/chat-text-size'

import { AppearanceSettings } from './appearance-settings'

// #40768: conversation text had no size control — only whole-UI zoom, which
// wastes vertical space on dense layouts. The row scales just the transcript
// and its captions, persisting like the other renderer-owned appearance prefs.

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

function chatTextSizeControls() {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <AppearanceSettings subpage="typography" />
    </QueryClientProvider>
  )

  const row = globalThis.document.getElementById('setting-field-appearance.chat-text-size')

  return {
    scale: (label: string) => within(row as HTMLElement).getByRole('button', { name: label })
  }
}

describe('Chat Text Size setting', () => {
  beforeEach(() => {
    window.localStorage.clear()
    act(() => setChatTextSize('100'))
  })

  afterEach(() => {
    cleanup()
    act(() => setChatTextSize('100'))
  })

  it('scales the conversation CSS vars from the row and persists the pick', () => {
    const { scale } = chatTextSizeControls()

    fireEvent.click(scale('150%'))

    expect($chatTextSize.get()).toBe('150')
    expect(cssVar('--conversation-text-font-size')).toBe('1.2188rem')
    expect(window.localStorage.getItem('hermes.desktop.chatTextSize.v1')).toBe('150')
  })

  it('returns to the styles.css defaults when 100% is picked again', () => {
    act(() => setChatTextSize('150'))
    const { scale } = chatTextSizeControls()

    fireEvent.click(scale('100%'))

    expect($chatTextSize.get()).toBe('100')
    expect(cssVar('--conversation-text-font-size')).toBe('')
    expect(window.localStorage.getItem('hermes.desktop.chatTextSize.v1')).toBeNull()
  })
})
