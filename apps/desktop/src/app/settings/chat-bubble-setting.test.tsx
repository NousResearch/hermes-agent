// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $chatBubbles, resetChatBubbles } from '@/store/chat-bubbles'
import { $userBubbleTransparency } from '@/store/user-bubble-transparency'

import { ChatBubbleSetting } from './chat-bubble-setting'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        appearance: {
          chatBubblesColor: { accent: 'Accent', neutral: 'Neutral', theme: 'Theme' },
          chatBubblesColorTitle: 'Color',
          chatBubblesCorners: { pill: 'Pill', round: 'Round', soft: 'Soft' },
          chatBubblesCornersTitle: 'Corners',
          chatBubblesDesc: 'Bubbles put your prompts on the right.',
          chatBubblesFillTitle: 'Fill',
          chatBubblesLayoutBubbles: 'Bubbles',
          chatBubblesLayoutDocument: 'Document',
          chatBubblesTitle: 'Chat Layout'
        }
      }
    }
  })
}))

const segment = (label: string) => screen.getByRole('button', { name: label })

describe('ChatBubbleSetting', () => {
  beforeEach(() => {
    resetChatBubbles()
    $userBubbleTransparency.set(100)
  })

  afterEach(cleanup)

  it('keeps the settings page short until the bubble layout is chosen', () => {
    render(<ChatBubbleSetting />)

    expect(screen.getByText('Chat Layout')).toBeTruthy()
    expect(screen.queryByText('Corners')).toBeNull()
    expect(screen.queryByText('Color')).toBeNull()
    expect(screen.queryByText('Fill')).toBeNull()
  })

  it('turns the layout on and exposes every knob in the panel', () => {
    render(<ChatBubbleSetting />)
    fireEvent.click(segment('Bubbles'))

    expect($chatBubbles.get().layout).toBe('bubbles')
    expect(segment('Bubbles').getAttribute('aria-pressed')).toBe('true')

    fireEvent.click(segment('Pill'))
    expect($chatBubbles.get().corners).toBe('pill')

    fireEvent.click(segment('Neutral'))
    expect($chatBubbles.get().tone).toBe('neutral')

    // The fill lever is the same state as the standalone Message Bubble row, so
    // one transparency governs both bubble sides instead of two that drift.
    fireEvent.change(screen.getByRole('slider', { name: 'Fill' }), { target: { value: '40' } })
    expect($userBubbleTransparency.get()).toBe(40)
  })

  it('goes back to the document layout and hides the panel again', () => {
    render(<ChatBubbleSetting />)
    fireEvent.click(segment('Bubbles'))
    fireEvent.click(segment('Document'))

    expect($chatBubbles.get().layout).toBe('document')
    expect(screen.queryByText('Corners')).toBeNull()
  })
})
