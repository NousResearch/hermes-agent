import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib'
import { CHAT_EMPTY_AREA } from '@/lib/chat-empty'
import { CHAT_WELCOME_AREA } from '@/lib/chat-welcome'

/**
 * Issue #95595 proposed-fix #3: the `messageComponents` map handed to
 * ThreadMessageList must keep its REFERENCE IDENTITY across a session switch.
 * If it re-minted, React would unmount/remount every visible message — async
 * re-rendered parts (shiki code blocks) collapse and re-expand, and the whole
 * thread visibly jumps on every tab switch.
 *
 * The memo deps are deliberately only the boolean "definedness" gates (the
 * callbacks themselves reach the composer through a ref), so a plain switch
 * — sessionId changing, callbacks unchanged — must not change the map.
 */
let lastComponents: unknown

vi.mock('@/components/assistant-ui/thread/list', () => ({
  ThreadMessageList: (props: { components: unknown; emptyPlaceholder?: React.ReactNode }) => {
    lastComponents = props.components

    return <>{props.emptyPlaceholder}</>
  }
}))

vi.mock('@/components/assistant-ui/thread/timeline', () => ({
  ThreadTimeline: () => null
}))

vi.mock('@/components/assistant-ui/thread/status', () => ({
  BackgroundResumeNotice: () => null,
  CenteredThreadSpinner: () => null
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: {
        thread: {
          restoreBody: 'restore body',
          restoreConfirm: 'Restore',
          restoreTitle: 'Restore this turn?'
        }
      },
      common: {
        cancel: 'Cancel',
        confirm: 'Confirm',
        done: 'Done',
        loading: 'Loading'
      }
    }
  })
}))

import { Thread } from './index'

const disposers: (() => void)[] = []

afterEach(() => {
  for (const dispose of disposers.splice(0)) {
    dispose()
  }
})

describe('Thread messageComponents identity across session switches', () => {
  it('does not re-mint messageComponents when only the session changes', () => {
    const { rerender } = render(<Thread sessionId="session-a" />)
    const first = lastComponents

    expect(first).toBeDefined()

    rerender(<Thread sessionId="session-b" />)

    // THE guard: a switch must keep the component map reference, so the
    // incoming transcript reconciles instead of remounting.
    expect(lastComponents).toBe(first)

    rerender(<Thread sessionId="session-c" />)

    expect(lastComponents).toBe(first)
  })

  it('keeps the map stable across a plain parent re-render', () => {
    const { rerender } = render(<Thread sessionId="session-a" />)
    const first = lastComponents

    rerender(<Thread sessionId="session-a" />)

    expect(lastComponents).toBe(first)
  })
})

describe('Thread empty-state routing', () => {
  it('replaces the draft Intro with a claimed chat welcome contribution', () => {
    disposers.push(
      registry.register({
        area: CHAT_WELCOME_AREA,
        data: { render: () => <div data-testid="welcome">draft welcome</div> },
        id: 'welcome'
      })
    )

    const { container } = render(
      <Thread cwd="/work/research" intro={{ personality: 'default', seed: 1 }} profile="lab" />
    )

    expect(screen.getByTestId('welcome').textContent).toBe('draft welcome')
    expect(container.querySelector('[data-slot="aui_intro"]')).toBeNull()
  })

  it('keeps session-backed empty transcripts on chat.empty', () => {
    let welcomeCalls = 0
    disposers.push(
      registry.register({
        area: CHAT_WELCOME_AREA,
        data: {
          render: () => {
            welcomeCalls += 1

            return <div>wrong surface</div>
          }
        },
        id: 'welcome'
      }),
      registry.register({
        area: CHAT_EMPTY_AREA,
        data: { render: () => <div data-testid="session-empty">session empty</div> },
        id: 'session-empty'
      })
    )

    render(<Thread sessionId="session-1" />)

    expect(screen.getByTestId('session-empty').textContent).toBe('session empty')
    expect(welcomeCalls).toBe(0)
  })
})
