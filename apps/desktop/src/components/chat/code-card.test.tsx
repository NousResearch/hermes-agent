import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { CodeCardBody } from './code-card'

describe('CodeCardBody', () => {
  it('wraps fenced pre instead of enabling horizontal scroll', () => {
    const { container } = render(
      <CodeCardBody>
        <pre>{'a'.repeat(400)}</pre>
      </CodeCardBody>
    )

    expect(container.firstElementChild?.className).toContain('overflow-x-hidden')
    expect(container.firstElementChild?.className).toContain('whitespace-pre-wrap')
    expect(container.firstElementChild?.className).toContain('[overflow-wrap:anywhere]')
    expect(container.firstElementChild?.className).not.toContain('overflow-x-auto')
  })
})
