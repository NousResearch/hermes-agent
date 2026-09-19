// Direction contract (lib/bidi): every content block carries an explicit dir
// from a whole-string majority vote, so a Persian block that OPENS with an
// English word still resolves RTL (first-strong resolution flipped such lines
// LTR and made them unreadable). ul/ol/blockquote/table vote over their whole
// subtree (markers, quote border, and column order follow the box), p/h/li
// and td/th vote per block (the CSS isolate override lets a voted dir beat
// the plaintext fallback), and inline code + directive chips carry dir="ltr"
// so they neither vote nor reorder. jsdom does not resolve direction, so the
// contract is asserted at the attribute level.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'
import { UserMessageText } from './user-message-text'

const createdAt = new Date('2026-06-01T00:00:00.000Z')
stubThreadEnvironment()

stubThreadViewportSize()

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'hi' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(text: string): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function Harness({ text }: { text: string }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistantMessage(text)],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('block-level direction chrome', () => {
  it('lists carry the voted direction so markers follow the content', async () => {
    render(<Harness text={'מקומות:\n\n1. חוף גורדון\n2. שוק הכרמל\n\n- פריט\n- item'} />)

    const item = await screen.findByText(/חוף גורדון/)

    expect(item.closest('ol')?.getAttribute('dir')).toBe('rtl')

    const bullet = await screen.findByText(/פריט/)

    expect(bullet.closest('ul')?.getAttribute('dir')).toBe('rtl')
  })

  it('blockquotes carry the voted direction so the border follows the content', async () => {
    render(<Harness text={'> ציטוט קצר בעברית'} />)

    const quote = await screen.findByText(/ציטוט קצר/)

    expect(quote.closest('blockquote')?.getAttribute('dir')).toBe('rtl')
  })

  it('inline code carries dir="ltr" so it does not vote in the resolution', async () => {
    render(<Harness text={'1. `npm install` מתקין תלויות'} />)

    const code = await screen.findByText('npm install')

    expect(code.tagName).toBe('CODE')
    expect(code.getAttribute('dir')).toBe('ltr')
    expect(code.closest('ol')?.getAttribute('dir')).toBe('rtl')
  })

  it('plain prose blocks carry the voted direction (not first-strong)', async () => {
    render(<Harness text={'שלום לכולם'} />)

    const paragraph = await screen.findByText(/שלום לכולם/)

    expect(paragraph.closest('p')?.getAttribute('dir')).toBe('rtl')
  })

  it('tables carry the voted direction so column order follows the content', async () => {
    render(<Harness text={'| نام | سن |\n| --- | --- |\n| سارا | ۱۹ |\n'} />)

    const cell = await screen.findByText(/سارا/)

    expect(cell.closest('table')?.getAttribute('dir')).toBe('rtl')
    expect(cell.closest('td')?.getAttribute('dir')).toBe('rtl')
  })

  it('header cells align to start instead of pinning left', async () => {
    render(<Harness text={'| نام | سن |\n| --- | --- |\n| سارا | ۱۹ |\n'} />)

    await screen.findByText(/سارا/)

    const th = document.querySelector('th')

    expect(th?.className).toMatch(/text-start/)
    expect(th?.className).not.toMatch(/text-left/)
  })
})

describe('user-bubble direction chrome', () => {
  it('inline code carries dir="ltr" so it does not vote or reorder', async () => {
    render(<UserMessageText text={'`npm install` را اجرا کن'} />)

    const code = await screen.findByText('npm install')

    expect(code.tagName).toBe('CODE')
    expect(code.getAttribute('dir')).toBe('ltr')
  })

  it('directive chips carry dir="ltr" so paths never vote', async () => {
    render(<UserMessageText text={'see @file:`apps/desktop/a b.ts` please'} />)

    const chip = document.querySelector('[data-slot="aui_directive-chip"]')

    expect(chip?.getAttribute('dir')).toBe('ltr')
  })
})

describe('majority vote beats first-strong', () => {
  it('a Persian paragraph opening with English stays rtl (the reported bug)', async () => {
    render(<Harness text={'Task Manager را باز کن و ادامه بده'} />)

    const paragraph = await screen.findByText(/ادامه بده/)

    expect(paragraph.closest('p')?.getAttribute('dir')).toBe('rtl')
  })

  it('an English paragraph opening with Persian stays ltr', async () => {
    render(<Harness text={'را بزن End task to see all the details here'} />)

    const paragraph = await screen.findByText(/details here/)

    expect(paragraph.closest('p')?.getAttribute('dir')).toBe('ltr')
  })

  it('headings vote over their whole text', async () => {
    render(<Harness text={'## Task Manager را باز کن و ادامه بده'} />)

    const heading = await screen.findByText(/ادامه بده/)

    expect(heading.closest('h2')?.getAttribute('dir')).toBe('rtl')
  })

  it('each list item votes for itself', async () => {
    render(<Harness text={'- Task Manager را باز کن و ادامه بده\n- Just an English item here'} />)

    const persianItem = await screen.findByText(/ادامه بده/)
    const englishItem = await screen.findByText('Just an English item here')

    expect(persianItem.closest('li')?.getAttribute('dir')).toBe('rtl')
    expect(englishItem.closest('li')?.getAttribute('dir')).toBe('ltr')
  })

  it('each user-bubble line votes for itself', async () => {
    render(<UserMessageText text={'See the details here\nTask Manager را باز کن و ادامه بده'} />)

    await screen.findByText(/ادامه بده/)

    const container = document.querySelector('[data-slot="aui_user-inline-text"]')
    const dirs = Array.from(container?.children ?? []).map(el => el.getAttribute('dir'))

    expect(dirs).toEqual(['ltr', 'rtl'])
  })
})
