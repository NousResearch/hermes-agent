/**
 * `::preview` with no usable `file` used to render NOTHING: the directive
 * parsed, a contribution claimed it, and the component returned `null`. That
 * is the exact failure mode the drop badge exists to end — the panel is gone
 * and the transcript looks as if the model never emitted anything.
 *
 * A claimed-but-empty render is indistinguishable from a dropped directive
 * from where the user sits, so it must leave a visible trace.
 */
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { InlinePreviewDirective } from './inline-preview-directive'

describe('InlinePreviewDirective with an unusable file attribute', () => {
  it.each([
    ['no file attribute at all', {}],
    ['an empty file attribute', { file: '' }],
    ['a misspelled attribute name', { fil: 'typo.html' }]
  ])('renders a visible fallback for %s', (_label, attrs) => {
    const { container } = render(<InlinePreviewDirective attrs={attrs} streaming={false} />)

    expect(container.textContent?.trim()).not.toBe('')
  })

  it('still renders nothing extra for a usable non-HTML target', () => {
    const { container } = render(<InlinePreviewDirective attrs={{ file: 'data.csv' }} streaming={false} />)

    expect(container.textContent?.trim()).not.toBe('')
  })
})
