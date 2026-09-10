import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

import { MarkdownTextContent } from './markdown-text'

/**
 * Reasoning is the model's scratchpad, and nothing in a scratchpad may be
 * promoted into app chrome.
 *
 * The reported case: mid-onboarding the model reminded itself which card came
 * next, wrote `::onboarding{step="look"}` inside its thinking block, and the
 * transcript mounted a second live accent picker in there — clickable, and
 * detached from the step the flow was actually on.
 */
afterEach(cleanup)

const DIRECTIVE = 'Next up is the colour card.\n\n::picker'

function registerPicker() {
  return registry.register({
    area: TRANSCRIPT_DIRECTIVE_AREA,
    data: { name: 'picker', render: () => <button type="button">Pick a colour</button> } satisfies TranscriptDirectiveContribution,
    id: 'test:picker',
    source: 'plugin:test'
  })
}

describe('scratchpad markdown', () => {
  it('leaves a transcript directive as text instead of mounting its card', () => {
    const dispose = registerPicker()

    try {
      const { container } = render(<MarkdownTextContent isRunning={false} scratchpad text={DIRECTIVE} />)

      expect(container.textContent).toContain('::picker')
      expect(screen.queryByRole('button')).toBeNull()
    } finally {
      dispose()
    }
  })

  it('still resolves the same directive on the answer path', async () => {
    const dispose = registerPicker()

    try {
      render(<MarkdownTextContent isRunning={false} text={DIRECTIVE} />)

      expect(await screen.findByRole('button', { name: 'Pick a colour' })).toBeTruthy()
    } finally {
      dispose()
    }
  })
})
