import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { CRONJOB_DELIVERY_RE } from './user-message'

import { Thread } from '.'

stubThreadEnvironment()
afterEach(cleanup)

// The exact wrapper cron/scheduler_delivery.py prepends when a scheduled job's
// output is delivered into the chat on the user role.
const frame = (name: string, body: string) =>
  `[Cronjob "${name}" output — scheduled job, not the user. Review it, act on anything that needs action, and summarize for the chat.]\n\n${body}`

// Scheduled cron deliveries render as a compact timeline notice with the
// job's output one click away, not as a user bubble. This pins both halves:
// the frame detection and the render.
describe('cronjob delivery detection', () => {
  it('captures the job name and the full multi-line output', () => {
    const m = CRONJOB_DELIVERY_RE.exec(frame('💬 每日主动聊天（Bot Chat）', 'line one\n\nline two'))

    expect(m?.[1]).toBe('💬 每日主动聊天（Bot Chat）')
    expect(m?.[2]).toBe('line one\n\nline two')
  })

  it('tolerates an empty output', () => {
    expect(CRONJOB_DELIVERY_RE.exec(frame('Daily Digest', ''))?.[2]).toBe('')
  })

  it('does not match prose that merely contains the frame', () => {
    expect(CRONJOB_DELIVERY_RE.test('look at [Cronjob "x" output — scheduled job, not the user.]')).toBe(false)
    expect(CRONJOB_DELIVERY_RE.test('Cronjob "x" output — scheduled job, not the user.] trailing')).toBe(false)
  })
})

it('renders a cron delivery as a compact notice, not a user bubble, with the output behind a disclosure', () => {
  const name = '💬 每日主动聊天（Bot Chat）'
  const body = '嘿，以后每天这个点我都来串个门、唠两句哈 😄'

  const messages = toChatMessages([{ role: 'user', content: frame(name, body), timestamp: 1 } as SessionMessage]).map(
    toRuntimeMessage
  )

  const { container } = render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )

  // Not a human prompt: no bubble and no raw wrapper text on screen.
  expect(container.querySelector('.composer-human-message')).toBeNull()
  expect(container.textContent).not.toContain('[Cronjob')

  // A compact "Scheduled job" notice names the job; the output is one click away.
  const note = container.querySelector('[data-slot="aui_cronjob-note"]')

  expect(note).toBeTruthy()

  if (!note) {
    throw new Error('cronjob notice not rendered')
  }

  expect(note.textContent).toContain(name)

  const details = note.querySelector('details')

  expect(details).toBeTruthy()
  expect(details?.open).toBe(false)
  expect(details?.textContent).toContain(body)

  fireEvent.click(details!.querySelector('summary')!)
  expect(details?.open).toBe(true)
})
