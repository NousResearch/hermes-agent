import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { UserMessageText } from '@/components/assistant-ui/thread/user-message-text'
import type { SessionMessage } from '@/types/hermes'

import { toChatMessages } from './hydration'
import { chatMessageText } from './parts'

afterEach(cleanup)

const note =
  '[Triggering message id: `1550380365858865156` — use as `message_id` for reply/react/pin via the discord tools.]'

// Regression for #114719: old stored rows still carry model-facing routing text.
describe('Discord history display', () => {
  it('renders the authored message and reply context without the routing envelope, without mutating stored rows', () => {
    const body = 'Please check `status`\n下一步是什么？'

    const replies = [
      '',
      '[Replying to: "first line\nsecond line"]\n\n',
      '[Replying to your previous message: "done"]\n\n'
    ]

    for (const reply of replies) {
      const content = `${reply}${note}\n\n${body}`
      const stored: SessionMessage[] = [{ id: 42, role: 'user', content, timestamp: 1 }]
      const original = structuredClone(stored)
      const [message] = toChatMessages(stored)
      const displayed = chatMessageText(message)
      const view = render(<UserMessageText text={displayed} />)

      expect(displayed).toBe(`${reply}${body}`)
      expect(view.container.textContent).not.toContain('Triggering message id:')
      expect(view.container.textContent).toContain('下一步是什么？')
      expect(message.rowId).toBe(42)
      expect(stored).toEqual(original)
      view.unmount()
    }
  })

  it('preserves routing-note examples in authored prose, quotations, code and assistant messages', () => {
    const examples = [
      `Explain this:\n\n${note}`,
      `> ${note}`,
      `\`\`\`text\n${note}\n\`\`\``,
      `[Replying to: "${note}"]\n\nThis is a quotation.`,
      '[Triggering message id: `123` — user-authored explanation.]\n\nKeep this.'
    ]

    for (const content of examples) {
      expect(toChatMessages([{ role: 'user', content }]).map(chatMessageText)).toEqual([content])
    }

    expect(toChatMessages([{ role: 'assistant', content: `${note}\n\nExplanation` }]).map(chatMessageText)).toEqual([
      `${note}\n\nExplanation`
    ])
  })
})
