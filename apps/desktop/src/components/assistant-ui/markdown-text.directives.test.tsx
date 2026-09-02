// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { resetDirectiveDiagnostics } from '@/lib/directive-diagnostics'
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

/**
 * The three silent-failure paths, through the real render tree. Each one used
 * to leave nothing behind but raw text in the transcript.
 */
describe('MarkdownTextContent directive diagnostics', () => {
  let warn: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    resetDirectiveDiagnostics()
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
  })

  afterEach(() => {
    cleanup()
    warn.mockRestore()
  })

  const logged = () => warn.mock.calls.map((call: unknown[]) => call.join(' ')).join('\n')

  it('warns when a directive-looking paragraph does not parse', () => {
    render(<MarkdownTextContent isRunning={false} text='::followup{p1="a"}}' />)

    expect(logged()).toContain('[transcript-directive]')
    expect(logged()).toContain('unexpected text after the closing brace')
  })

  it('warns when a parsed directive has no plugin, naming what is registered', async () => {
    const dispose = registerFollowup()

    try {
      render(<MarkdownTextContent isRunning={false} text='::tasks{id="1"}' />)

      await vi.waitFor(() => expect(logged()).toContain('no plugin claims "::tasks"'))
      expect(logged()).toContain('followup')
    } finally {
      dispose()
    }
  })

  it('warns when a claimed directive throws, and keeps the message alive', async () => {
    const dispose = registry.register({
      id: 'test:exploding',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      source: 'plugin:test',
      data: {
        name: 'exploding',
        render: () => {
          throw new Error('widget blew up')
        }
      } satisfies TranscriptDirectiveContribution
    })

    // The boundary logs the raw React error too; only our line is asserted.
    const error = vi.spyOn(console, 'error').mockImplementation(() => {})

    try {
      const { container } = render(<MarkdownTextContent isRunning={false} text={'Trước.\n\n::exploding{a="b"}'} />)

      await vi.waitFor(() => expect(logged()).toContain('"::exploding" (test:exploding) failed to render'))
      expect(logged()).toContain('widget blew up')
      // The rest of the message must survive the crashed widget.
      expect(container.textContent).toContain('Trước.')
    } finally {
      error.mockRestore()
      dispose()
    }
  })

  it('stays quiet for a healthy directive and for ordinary prose', async () => {
    const dispose = registerFollowup()

    try {
      render(<MarkdownTextContent isRunning={false} text={`Bình thường.\n\n${FOLLOWUP}`} />)

      await screen.findByTestId('followup-panel')
      expect(logged()).not.toContain('[transcript-directive]')
    } finally {
      dispose()
    }
  })

  it('stays quiet while a directive is still streaming in', () => {
    // Every prefix of a directive is malformed; warning here would fire on
    // essentially every token of every directive ever emitted.
    render(<MarkdownTextContent isRunning={true} text='::followup{p1="Bắn m' />)

    expect(logged()).not.toContain('[transcript-directive]')
  })
})

/**
 * `::preview` is the directive that ships in core, and its `file` attribute is
 * a path — globs and underscores are ordinary filenames. Same bug class as the
 * Follow-up regression, so pin it through the real render path too.
 */
describe('MarkdownTextContent preview directive', () => {
  afterEach(cleanup)

  function registerPreviewProbe(seen: string[]) {
    return registry.register({
      id: 'test:preview',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      source: 'plugin:test',
      data: {
        name: 'preview',
        render: ({ attrs }) => {
          seen.push(attrs.file ?? '')

          return <div data-testid="preview-probe">{attrs.file}</div>
        }
      } satisfies TranscriptDirectiveContribution
    })
  }

  it.each([
    ['underscore', 'my_report.html'],
    ['asterisk glob', 'wt-*/demo.html'],
    ['tilde', 'backup~1.html'],
    ['brackets', 'notes[1].html'],
    ['unicode', 'báo-cáo.html']
  ])('renders a %s filename intact', async (_label, file) => {
    const seen: string[] = []
    const dispose = registerPreviewProbe(seen)

    try {
      const { container } = render(<MarkdownTextContent isRunning={false} text={`::preview{file="${file}"}`} />)

      await screen.findByTestId('preview-probe')

      expect(seen).toEqual([file])
      expect(container.textContent).not.toContain('::preview')
    } finally {
      dispose()
    }
  })
})
