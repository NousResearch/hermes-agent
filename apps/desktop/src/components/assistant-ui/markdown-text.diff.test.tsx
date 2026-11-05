import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

const RAW_DIFF = [
  '# Pending skill write b4643400',
  '',
  "## op 1/1: patch 'systematic-debugging' SKILL.md",
  '',
  '--- a/SKILL.md',
  '+++ b/SKILL.md',
  '@@ -44,1 +44,2 @@',
  ' before',
  '-old instruction',
  '+new instruction',
  ''
].join('\n')

afterEach(() => cleanup())

describe('MarkdownTextContent unified diffs', () => {
  it('renders a bare skill diff through the scrollable diff panel', () => {
    const { container } = render(<MarkdownTextContent isRunning={false} text={RAW_DIFF} />)
    const panel = container.querySelector('[data-slot="file-diff-panel"]')

    expect(panel).not.toBeNull()
    expect(panel?.textContent).toContain('old instruction')
    expect(panel?.textContent).toContain('new instruction')
    expect(panel?.className).toContain('overflow-auto')
    expect(panel?.querySelector('.absolute.inset-0')).toBeNull()
    expect(container.querySelector('[data-slot="code-card"]')).toBeNull()
  })

  it('does not wrap an already fenced diff twice', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={`\`\`\`diff\n${RAW_DIFF.split('\n').slice(4).join('\n')}\n\`\`\``} />
    )

    expect(container.querySelectorAll('[data-slot="file-diff-panel"]')).toHaveLength(1)
  })

  it('leaves a thematic rule without a unified hunk untouched', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={'--- a/SKILL.md\n+++ b/SKILL.md\nOrdinary text'} />
    )

    expect(container.querySelector('[data-slot="file-diff-panel"]')).toBeNull()
  })
})
