import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { clearArtifactRegistry } from '@/store/artifacts'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { MarkdownTextContent } from './markdown-text'

function fenced(language: string, body: string): string {
  return `\`\`\`${language}\n${body}\n\`\`\`\n`
}

// A fenced code block renders a hover-revealed word-wrap switch next to the
// copy button; toggling it flags the card so styles.css reflows long lines
// instead of scrolling horizontally.
describe('MarkdownTextContent word-wrap toggle', () => {
  beforeEach(() => {
    $activeSessionId.set('session-wrap')
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    clearArtifactRegistry()
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    clearArtifactRegistry()
    window.localStorage.clear()
  })

  it('toggles word wrap on a code block', () => {
    const { container } = render(<MarkdownTextContent isRunning={false} text={fenced('js', 'const x = 1')} />)

    const card = () => container.querySelector('[data-slot="code-card"]')

    expect(card()).not.toBeNull()
    expect(card()?.hasAttribute('data-wrap')).toBe(false)

    fireEvent.click(screen.getByRole('switch', { name: 'Toggle word wrap' }))

    expect(card()?.hasAttribute('data-wrap')).toBe(true)
  })
})
