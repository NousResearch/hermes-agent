// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

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
