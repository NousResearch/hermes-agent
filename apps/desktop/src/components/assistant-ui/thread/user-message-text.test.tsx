import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { referenceRe, WIRE_REFERENCE_KINDS } from '@/components/assistant-ui/reference-kinds'

import { UserMessageText } from './user-message-text'

afterEach(cleanup)

/**
 * A sent reference must render as the chip the composer showed. These cover the
 * seam where that used to break: the value's quoting is directive syntax, and a
 * surface that reads it as markdown splits one reference into two wrong things.
 */
describe('a sent reference renders as the chip the composer showed', () => {
  it('chips a backtick-quoted @url: instead of splitting it into code', () => {
    render(
      <UserMessageText text="@url:`https://github.com/NousResearch/hermes-agent/pull/74790` urls lose formatting" />
    )

    expect(screen.queryByTitle('https://github.com/NousResearch/hermes-agent/pull/74790')).not.toBeNull()
    // The whole reference is one node — no bare `@url:` text left behind.
    expect(document.body.textContent).not.toContain('@url:')
  })

  it('chips a backtick-quoted @file: path with spaces', () => {
    render(<UserMessageText text="see @file:`apps/desktop/my notes.md` please" />)

    expect(screen.queryByTitle('apps/desktop/my notes.md')).not.toBeNull()
    expect(document.body.textContent).not.toContain('@file:')
  })

  it('chips every kind that travels in message text', () => {
    // The guard against WIRE_REFERENCE_KINDS and the pattern's own alternation
    // drifting apart: add a kind to one and this fails until both agree.
    for (const kind of WIRE_REFERENCE_KINDS) {
      expect(`@${kind}:\`some value\``.match(referenceRe()), kind).toHaveLength(1)
    }
  })

  it('still renders a genuine code span as code', () => {
    render(<UserMessageText text="run `npm test` first" />)

    const code = document.querySelector('[data-slot="aui_user-inline-code"]')

    expect(code?.textContent).toBe('npm test')
  })

  it('renders code and a reference side by side', () => {
    render(<UserMessageText text="run `npm test` on @file:`apps/desktop/a b.ts` now" />)

    expect(document.querySelector('[data-slot="aui_user-inline-code"]')?.textContent).toBe('npm test')
    expect(screen.queryByTitle('apps/desktop/a b.ts')).not.toBeNull()
  })

  it('leaves a fenced block alone', () => {
    render(<UserMessageText text={'before\n```ts\nconst x = 1\n```\nafter'} />)

    const fence = document.querySelector('[data-slot="aui_user-fence"]')
    expect(fence?.textContent).toBe('const x = 1\n')
    expect(fence?.className).toContain('overflow-x-hidden')
    expect(fence?.className).toContain('whitespace-pre-wrap')
    expect(fence?.className).not.toContain('overflow-x-auto')
  })
})
