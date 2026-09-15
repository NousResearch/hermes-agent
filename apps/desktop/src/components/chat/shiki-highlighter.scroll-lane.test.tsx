import { cleanup, render } from '@testing-library/react'
import { type ComponentProps } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { SyntaxHighlighter } from './shiki-highlighter'

// A fenced code block renders through CodeCard, whose scroller spans the full
// width of the card — so the card's right edge IS the scrollbar's lane, and
// `.scrollbar-overlay` gives that lane back to the platform (~15px macOS
// classic with a mouse attached, ~17px Windows) instead of the app's themed
// 4px gutter. The copy control floats over the same corner, so two things have
// to stay true together: the scroller keeps the platform's lane, and the
// control's inset clears it. At 6px the button landed on the bar it sits
// above, at the exact height the thumb occupies when scrolled to the top.
afterEach(cleanup)

const Pre = ({ children, ...props }: ComponentProps<'pre'>) => <pre {...props}>{children}</pre>
const Code = ({ children, ...props }: ComponentProps<'code'>) => <code {...props}>{children}</code>

describe('code-block copy control', () => {
  it('reserves the scrollbar lane its card hands to the platform', () => {
    const { container } = render(
      <SyntaxHighlighter code={'const a = 1\n'.repeat(40)} components={{ Code, Pre }} defer language="ts" />
    )

    const card = container.querySelector('[data-slot="code-card"]')
    const scroller = card?.querySelector('.scrollbar-overlay')
    const copy = card?.querySelector('button')

    // The lane is the platform's, so it is wider than the themed gutter the
    // 6px inset was tuned against...
    expect(scroller).toBeTruthy()

    // ...and the control has to reserve it. 16px clears a 15px macOS lane with
    // its own box and puts the icon 20px out, clear of the widest native bar.
    expect(copy?.className).toContain('right-4')
  })
})
