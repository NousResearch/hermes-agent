import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

// The RENDERER half of the view-mode contract: when LocalFilePreview lands on
// the 'rendered' mode for a .md file (see preview-view-mode.test.ts), the user
// must see formatted Markdown. This exercises the REAL Streamdown pipeline
// through MarkdownPreview's own component map — headings, lists, links,
// inline/fenced code, tables — plus the safety floor: no script execution.
import { MarkdownPreview } from './preview-file'

const DOC = [
  '# Release notes',
  '',
  'See the [migration guide](https://example.com/guide) first.',
  '',
  '- one',
  '- two',
  '',
  '1. first',
  '2. second',
  '',
  'Use `npm run build` locally.',
  '',
  '```ts',
  'export const answer = 42',
  '```',
  '',
  '| Stage | Status |',
  '| ----- | ------ |',
  '| build | green  |'
].join('\n')

function renderDoc() {
  return render(<MarkdownPreview text={DOC} />)
}

describe('MarkdownPreview (rendered file view)', () => {
  afterEach(cleanup)

  it('renders structure semantically', () => {
    const { container } = renderDoc()

    expect(screen.getByRole('heading', { level: 1, name: 'Release notes' })).toBeTruthy()
    expect(screen.getByText('migration guide').closest('a')).toHaveProperty('href')
    expect(container.querySelector('ul')?.children.length).toBe(2)
    expect(container.querySelector('ol')?.children.length).toBe(2)
    expect(container.querySelector('table')).not.toBeNull()
    // Inline code and fenced code both present.
    expect(screen.getByText('npm run build')).toBeTruthy()
    expect(screen.getByText(/answer = 42/)).toBeTruthy()
  })

  it('never executes or emits scripts from document content', () => {
    const { container } = render(
      <MarkdownPreview
        text={'# Title\n\n<script>window.__pwned = true</script>\n\n<img src=x onerror="window.__pwned = true">'}
      />
    )

    expect((window as unknown as Record<string, unknown>).__pwned).toBeUndefined()
    expect(container.querySelectorAll('script').length).toBe(0)
  })
})
