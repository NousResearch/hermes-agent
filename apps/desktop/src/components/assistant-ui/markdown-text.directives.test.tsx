// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

import { MarkdownTextContent } from './markdown-text'

/**
 * End-to-end for the reported bug: a Follow-up directive rendered as raw
 * `::followup{...}*` text instead of the plugin's panel.
 *
 * This drives the FULL render path — preprocessMarkdown -> remend ->
 * Streamdown -> paragraph -> TranscriptDirectiveLeaf -> contribution — so it
 * fails if the repair pass corrupts the directive again, whatever the cause.
 */

const FOLLOWUP =
  '::followup{p1="Bắn một đơn PRINTHUB canary mới lên Sheet và theo trọn vòng tới tracking" ' +
  'p2="Đối chiếu chênh giá 23.89 và 11.89 của đơn 550728" ' +
  'p3="Dọn các worktree wt-* đã merge để tránh lạc commit" ' +
  'p4="Thêm log có ID job và đơn cho nhịp kéo Sheet để truy vết từng lượt"}'

function registerFollowup() {
  return registry.register({
    id: 'test:followup',
    area: TRANSCRIPT_DIRECTIVE_AREA,
    source: 'plugin:test',
    data: {
      name: 'followup',
      render: ({ attrs }) => (
        <div data-testid="followup-panel">
          {Object.entries(attrs)
            .filter(([key]) => /^p\d+$/.test(key))
            .map(([key, value]) => (
              <button key={key} type="button">
                {value}
              </button>
            ))}
        </div>
      )
    } satisfies TranscriptDirectiveContribution
  })
}

describe('MarkdownTextContent transcript directives', () => {
  afterEach(cleanup)

  it('renders the Follow-up panel for a prompt containing an unpaired asterisk', async () => {
    const dispose = registerFollowup()

    try {
      const { container } = render(<MarkdownTextContent isRunning={false} text={`Xong.\n\n${FOLLOWUP}`} />)

      const panel = await screen.findByTestId('followup-panel')

      expect(panel).toBeTruthy()
      // The prompt that triggered the bug must survive intact and clickable.
      expect(screen.getByRole('button', { name: 'Dọn các worktree wt-* đã merge để tránh lạc commit' })).toBeTruthy()
      // And the raw directive must not be visible anywhere in the transcript.
      expect(container.textContent).not.toContain('::followup')
    } finally {
      dispose()
    }
  })

  it('renders the panel while the message is still streaming', async () => {
    const dispose = registerFollowup()

    try {
      const { container } = render(<MarkdownTextContent isRunning={true} text={FOLLOWUP} />)

      expect(await screen.findByTestId('followup-panel')).toBeTruthy()
      expect(container.textContent).not.toContain('::followup')
    } finally {
      dispose()
    }
  })

  it('leaves an unclaimed directive as text', () => {
    const { container } = render(<MarkdownTextContent isRunning={false} text='::nobodyclaims{p1="x"}' />)

    expect(container.textContent).toContain('::nobodyclaims')
  })
})
