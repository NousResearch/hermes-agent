// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { parseMarkdownIntoBlocksCached } from '@/lib/markdown-blocks'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import {
  createTranscriptDirectiveCodec,
  finalizeTranscriptDirectiveBlocks,
  protectTranscriptDirectiveBlocks,
  TRANSCRIPT_DIRECTIVE_AREA,
  type TranscriptDirectiveContribution
} from '@/lib/transcript-directives'

import { MarkdownTextContent } from './markdown-text'
import { paragraphPlainText, TranscriptDirectiveLeaf } from './transcript-directive'

describe('paragraphPlainText', () => {
  it('passes through a plain string', () => {
    expect(paragraphPlainText('::tasks')).toBe('::tasks')
  })

  it('joins an all-string child array (streamed text chunks)', () => {
    expect(paragraphPlainText(['::preview{file=', '"a.html"}'])).toBe('::preview{file="a.html"}')
  })

  it('disqualifies paragraphs with element children', () => {
    expect(paragraphPlainText(['::tasks ', <b key="x">bold</b>])).toBeNull()
    expect(paragraphPlainText(null)).toBeNull()
    expect(paragraphPlainText([])).toBeNull()
  })
})

describe('TranscriptDirectiveLeaf', () => {
  afterEach(cleanup)

  const contribution = (over?: Partial<TranscriptDirectiveContribution>) =>
    registry.register({
      id: 'test:demo',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      source: 'plugin:test',
      data: {
        name: 'demo',
        render: ({ attrs }) => <div data-testid="demo-card">{attrs.label ?? 'demo'}</div>,
        ...over
      } satisfies TranscriptDirectiveContribution
    })

  it('renders the registered component for a claimed directive', () => {
    const dispose = contribution()

    try {
      render(<TranscriptDirectiveLeaf text='::demo{label="hi"}' />)
      expect(screen.getByTestId('demo-card').textContent).toBe('hi')
    } finally {
      dispose()
    }
  })

  it('renders nothing for an unclaimed directive', () => {
    const { container } = render(<TranscriptDirectiveLeaf text="::nobody-home" />)

    expect(container.firstChild).toBeNull()
  })

  it('renders nothing for plain prose', () => {
    const { container } = render(<TranscriptDirectiveLeaf text="just some text" />)

    expect(container.firstChild).toBeNull()
  })

  it('contains a throwing plugin render to its own boundary', () => {
    const dispose = contribution({
      render: () => {
        throw new Error('plugin bug')
      }
    })

    try {
      render(<TranscriptDirectiveLeaf text="::demo" />)
      // The chip fallback renders the contribution id, not a dead subtree.
      expect(screen.getByRole('button')).toBeTruthy()
    } finally {
      dispose()
    }
  })
})

describe('TranscriptDirectiveCodec', () => {
  it('uses fixed-length Markdown-inert tokens and round-trips Unicode sources', () => {
    const codec = createTranscriptDirectiveCodec()
    const sources = ['::link', '::link{title="naïve ☃ ~~ __ [x]"}']
    const tokens = sources.map(source => codec.encode(source))

    for (const token of tokens) {
      expect(token).toMatch(/^hermestranscriptdirectivev1[A-F0-9]{40}$/u)
    }

    expect(tokens[0]?.length).toBe(tokens[1]?.length)
    expect(codec.encode(sources[0])).toBe(tokens[0])
    expect(codec.decode(tokens[0] ?? '')).toBe(sources[0])
    expect(codec.decode(tokens[1] ?? '')).toBe(sources[1])
  })

  it('caps each surface codec at 256 distinct directive sources', () => {
    const codec = createTranscriptDirectiveCodec()

    for (let index = 0; index < 256; index += 1) {
      expect(codec.encode(`::link{key="${index}"}`)).not.toBeNull()
    }

    expect(codec.encode('::link{key="overflow"}')).toBeNull()
  })

  it('fails closed when secure random values are unavailable', () => {
    const cryptoSource = globalThis.crypto
    vi.stubGlobal('crypto', undefined)

    try {
      expect(createTranscriptDirectiveCodec().encode('::link{key="x"}')).toBeNull()
    } finally {
      vi.stubGlobal('crypto', cryptoSource)
    }
  })

  it('fails closed when secure random generation throws', () => {
    const cryptoSource = globalThis.crypto
    vi.stubGlobal('crypto', {
      getRandomValues() {
        throw new Error('synthetic rng failure')
      }
    })

    try {
      expect(createTranscriptDirectiveCodec().encode('::link{key="x"}')).toBeNull()
    } finally {
      vi.stubGlobal('crypto', cryptoSource)
    }
  })

  it('remints a finalized raw token into a separate claim codec', () => {
    const protectionCodec = createTranscriptDirectiveCodec()
    const claimCodec = createTranscriptDirectiveCodec()
    const source = '::link{key="x"}'
    const protectedToken = protectionCodec.encode(source)

    if (!protectedToken) {
      throw new Error('Expected the protection codec to encode a valid directive')
    }

    const finalized = finalizeTranscriptDirectiveBlocks(
      protectedToken,
      parseMarkdownIntoBlocksCached,
      protectionCodec,
      claimCodec
    )

    expect(finalized).not.toBe(protectedToken)
    expect(protectionCodec.decode(finalized)).toBeNull()
    expect(claimCodec.decode(finalized)).toBe(source)
  })
})

describe('protectTranscriptDirectiveBlocks', () => {
  it('returns markdown without directive candidates without invoking the block parser', () => {
    const source = 'plain CRLF prose\r\nwith no directive candidate'
    const codec = createTranscriptDirectiveCodec()

    expect(
      protectTranscriptDirectiveBlocks(
        source,
        () => {
          throw new Error('block parser should not run')
        },
        codec
      )
    ).toBe(source)
  })

  it('returns the original Markdown when the block parser throws', () => {
    const source = '::link{key="x"}'

    expect(
      protectTranscriptDirectiveBlocks(
        source,
        () => {
          throw new Error('synthetic parser failure')
        },
        createTranscriptDirectiveCodec()
      )
    ).toBe(source)
  })

  it.each([
    ['CDATA-like markup', '<div><![CDATA[ ><script>foo]]></div>\n\n::link{key="after-cdata"}'],
    ['a malformed closing tag', '<div></div "><script>foo ">\n\n::link{key="after-malformed-close"}'],
    ['a bogus declaration', '<div><!foo "><script>foo" > </div>\n\n::link{key="after-bogus-declaration"}']
  ])('does not mint a directive after %s', (_label, source) => {
    expect(protectTranscriptDirectiveBlocks(source, parseMarkdownIntoBlocksCached, createTranscriptDirectiveCodec())).toBe(
      source
    )
  })

  it('preserves original CRLF bytes when directive-looking prose is not protected', () => {
    const source = 'alpha\r\n::link[1]\r\nomega'

    expect(protectTranscriptDirectiveBlocks(source, parseMarkdownIntoBlocksCached, createTranscriptDirectiveCodec())).toBe(
      source
    )
  })

  it('preserves CRLF around a protected standalone directive', () => {
    const source = '::link{key="x"}\r\n\r\nomega'

    const protectedValue = protectTranscriptDirectiveBlocks(
      source,
      parseMarkdownIntoBlocksCached,
      createTranscriptDirectiveCodec()
    )

    expect(protectedValue).toMatch(/^hermestranscriptdirectivev1[A-F0-9]{40}\r\n\r\nomega$/u)
  })
})

describe('MarkdownTextContent transcript directives', () => {
  afterEach(cleanup)

  it('preserves a URL attribute through markdown preprocessing', async () => {
    const dispose = registry.register({
      id: 'test:link',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      source: 'plugin:test',
      data: {
        name: 'link',
        render: ({ attrs }) => <div data-testid="link-card">{attrs.url}</div>
      } satisfies TranscriptDirectiveContribution
    })

    try {
      render(
        <MarkdownTextContent
          isRunning={false}
          text='::link{title="Hermes Desktop Plugin SDK" url="https://hermes-agent.nousresearch.com/docs/developer-guide/desktop-plugin-sdk" desc="Die offizielle Primärquelle für native Erweiterungen und Host-Schnittstellen."}'
        />
      )

      expect((await screen.findByTestId('link-card')).textContent).toBe(
        'https://hermes-agent.nousresearch.com/docs/developer-guide/desktop-plugin-sdk'
      )
    } finally {
      dispose()
    }
  })
})

describe('Link-fix blocker regressions', () => {
  afterEach(cleanup)

  const registerLink = () =>
    registry.register({
      id: 'test:link-regression',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      source: 'plugin:test',
      data: {
        name: 'link',
        render: ({ attrs }) => <div data-testid="link-card">{JSON.stringify(attrs)}</div>
      } satisfies TranscriptDirectiveContribution
    })

  const renderLink = (text: string, isRunning = false) =>
    render(<MarkdownTextContent isRunning={isRunning} text={text} />)

  const mintForeignToken = () => {
    const token = createTranscriptDirectiveCodec().encode('::link{url="https://x.test"}')

    if (!token) {
      throw new Error('Expected the foreign codec to encode a valid directive')
    }

    return token
  }

  it('keeps a minted but unclaimed directive as its original source text', async () => {
    const source = '::not-registered{key="x"}'

    renderLink(source)
    await waitFor(() => expect(globalThis.document.body.textContent).toContain(source))
    expect(screen.queryByTestId('link-card')).toBeNull()
  })

  it('does not claim an authentic token minted by another surface codec', async () => {
    const rawLookalike = mintForeignToken()

    const dispose = registerLink()

    try {
      renderLink(rawLookalike)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain(rawLookalike))
      expect(screen.queryByTestId('link-card')).toBeNull()
    } finally {
      dispose()
    }
  })

  it('leaves directive-looking lines to the markdown boundary instead of encoding them generically', () => {
    const source = '::link{label="plain"}'

    expect(preprocessMarkdown(source)).toBe(source)
  })

  it('keeps a directive followed by another line as ordinary prose', async () => {
    const dispose = registerLink()
    const source = '::link\ncontinued'

    try {
      renderLink(source)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain('::link'))
      expect(globalThis.document.body.textContent).toContain('continued')
      expect(globalThis.document.body.textContent).not.toContain('hermestranscriptdirectivev1')
      expect(screen.queryByTestId('link-card')).toBeNull()
    } finally {
      dispose()
    }
  })

  it.each([
    ['four-space indented code', '    ::link{url="https://x.test"}'],
    ['tab-indented code', '\t::link{url="https://x.test"}'],
    ['closed fenced code', '```\n::link{url="https://x.test"}\n```'],
    ['incomplete fenced code', '```\n::link{url="https://x.test"}'],
    ['list item', '- ::link{url="https://x.test"}'],
    ['blockquote', '> ::link{url="https://x.test"}'],
    ['HTML block', '<div>\n::link{url="https://x.test"}\n</div>']
  ])('keeps %s ordinary and never claims a directive', async (_label, text) => {
    const dispose = registerLink()

    try {
      renderLink(text)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain('::link'))
      expect(globalThis.document.body.textContent).not.toContain('hermestranscriptdirectivev1')
      expect(screen.queryByTestId('link-card')).toBeNull()
    } finally {
      dispose()
    }
  })

  it.each([
    ['loose list continuation', '- item\n\n  ::link{key="x"}', 'item'],
    ['loose blockquote continuation', '> item\n>\n> ::link{key="x"}', 'item'],
    ['citation normalization', '::link[1]', '::link'],
    [
      'closed prose-like fence',
      '```text\nalpha prose sentence\nbeta prose sentence\n::link{key="x"}\ngamma prose sentence\n```',
      'alpha prose sentence'
    ],
    [
      'incomplete prose-like fence',
      '```text\nalpha prose sentence\nbeta prose sentence\n::link{key="x"}\ngamma prose sentence',
      'alpha prose sentence'
    ],
    ['triple-backtick noise', '```::link{key="x"}```', '::link']
  ])('does not claim a directive produced by preprocessing from %s', async (_label, text, visibleText) => {
    const dispose = registerLink()

    try {
      renderLink(text)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain(visibleText))
      expect(screen.queryByTestId('link-card')).toBeNull()
    } finally {
      dispose()
    }
  })

  it.each([
    ['backtick scrubbing', '<```div>\n\n::link{key="post-backtick-html"}', 'post-backtick-html'],
    [
      'preview-marker stripping',
      '<[Preview:x](#preview/foo)div>\n\n::link{key="post-preview-html"}',
      'post-preview-html'
    ]
  ])('does not claim a raw directive after %s synthesizes HTML', async (_label, text, visibleText) => {
    const dispose = registerLink()

    try {
      renderLink(text)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain(visibleText))
      expect(screen.queryByTestId('link-card')).toBeNull()
      expect(globalThis.document.body.textContent).not.toContain('hermestranscriptdirectivev1')
    } finally {
      dispose()
    }
  })

  it.each([
    ['tab', '<div\tclass="directive-html-context">\n\n::link{key="inside-tab-html"}\n\n</div>', 'inside-tab-html'],
    [
      'form feed',
      '<div\fclass="directive-html-context">\n\n::link{key="inside-form-feed-html"}\n\n</div>',
      'inside-form-feed-html'
    ]
  ])('does not claim a directive inside raw HTML with %s attribute whitespace', async (_label, text, visibleText) => {
    const dispose = registerLink()

    try {
      renderLink(text)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain(visibleText))
      expect(screen.queryByTestId('link-card')).toBeNull()
      expect(globalThis.document.body.textContent).not.toContain('hermestranscriptdirectivev1')
    } finally {
      dispose()
    }
  })

  it.each([
    [
      'a closing tag embedded in an HTML comment',
      '<div><!-- </div> -->\n\n::link{key="inside-commented-close"}\n\n</div>',
      'inside-commented-close'
    ],
    [
      'a closing tag embedded in script raw text',
      '<div><script>const fake = "</div>"</script>\n\n::link{key="inside-script-fake-close"}\n\n</div>',
      'inside-script-fake-close'
    ]
  ])('keeps HTML context closed against %s', async (_label, text, visibleText) => {
    const dispose = registerLink()

    try {
      renderLink(text)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain(visibleText))
      expect(screen.queryByTestId('link-card')).toBeNull()
    } finally {
      dispose()
    }
  })

  it.each([
    ['a closed element', '<div>ordinary HTML</div>\n\n::link{key="after-closed-html"}', 'after-closed-html'],
    ['a void element', '<br>\n\n::link{key="after-void-html"}', 'after-void-html'],
    ['a closed comment', '<!-- ordinary comment -->\n\n::link{key="after-comment"}', 'after-comment']
  ])('keeps a later directive ordinary after %s', async (_label, text, expectedKey) => {
    const dispose = registerLink()

    try {
      renderLink(text)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain(expectedKey))
      expect(screen.queryByTestId('link-card')).toBeNull()
      expect(globalThis.document.body.textContent).not.toContain('hermestranscriptdirectivev1')
    } finally {
      dispose()
    }
  })

  it('round-trips markdown markers, links, punctuation, percent escapes, and Unicode attributes', async () => {
    const dispose = registerLink()

    const source =
      '::link{title="~~del __strong__ [label](https://example.test/a?x=1%2520)" url="https://example.test/a_(b)?x=1%2F2%25&y=[z]" unicode="naïve ☃"}'

    const expected = {
      title: '~~del __strong__ [label](https://example.test/a?x=1%2520)',
      url: 'https://example.test/a_(b)?x=1%2F2%25&y=[z]',
      unicode: 'naïve ☃'
    }

    try {
      renderLink(source)
      expect((await screen.findByTestId('link-card')).textContent).toBe(JSON.stringify(expected))
    } finally {
      dispose()
    }
  })

  it('does not treat HTML-looking attributes in a protected directive as raw HTML context', async () => {
    const dispose = registerLink()
    const source = '::link{key="<b>"}\n\n::link{key="second"}'

    try {
      renderLink(source)
      const cards = await screen.findAllByTestId('link-card')

      expect(cards.map(card => card.textContent)).toEqual([
        JSON.stringify({ key: '<b>' }),
        JSON.stringify({ key: 'second' })
      ])
    } finally {
      dispose()
    }
  })

  it.each([
    ['noncanonical lowercase hex', (token: string) => token.replace(/[A-F]/u, value => value.toLowerCase())],
    ['malformed non-hex suffix', (token: string) => `${token}G`],
    ['oversized payload', (token: string) => `${token}${'A'.repeat(5000)}`]
  ])('passes an actual-format %s token through unchanged', async (_label, mutateToken) => {
    const rawToken = mutateToken(mintForeignToken())
    const dispose = registerLink()

    try {
      renderLink(rawToken)
      await waitFor(() => expect(globalThis.document.body.textContent).toContain(rawToken))
      expect(screen.queryByTestId('link-card')).toBeNull()
    } finally {
      dispose()
    }
  })

  it('transitions from a partial directive to the card when streaming completes', async () => {
    const dispose = registerLink()
    const partial = '::link{url="https://x.test/path'
    const complete = '::link{url="https://x.test/path"}'

    try {
      const view = renderLink(partial, true)
      expect(screen.queryByTestId('link-card')).toBeNull()
      view.rerender(<MarkdownTextContent isRunning={false} text={complete} />)
      expect((await screen.findByTestId('link-card')).textContent).toBe(JSON.stringify({ url: 'https://x.test/path' }))
    } finally {
      dispose()
    }
  })
})
