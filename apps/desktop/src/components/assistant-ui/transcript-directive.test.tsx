// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'

import { paragraphPlainText, TranscriptDirectiveLeaf, useResolvedParagraph } from './transcript-directive'

// Both at module scope: the compiler hoists a component out of the test body,
// which would strand a closure over a `let` declared in there.
const mounts = { count: 0 }

const MountCounter: TranscriptDirectiveContribution['render'] = ({ streaming }) => {
  useEffect(() => {
    mounts.count += 1
  }, [])

  return <div data-testid="demo-card">{streaming ? 'live' : 'settled'}</div>
}

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

  describe('useResolvedParagraph', () => {
    const resolve = (text: string) => {
      let out: ReturnType<typeof useResolvedParagraph> = null

      function Probe() {
        out = useResolvedParagraph(text)

        return null
      }

      render(<Probe />)

      return out
    }

    it('splits a sentence the model ended with a directive', () => {
      const dispose = contribution()

      try {
        expect(resolve('Pick a color. ::demo{label="hi"}')).toEqual([
          { kind: 'prose', text: 'Pick a color. ' },
          { kind: 'directive', source: '::demo{label="hi"}' }
        ])
      } finally {
        dispose()
      }
    })

    it('leaves a whole-paragraph directive as the card alone', () => {
      const dispose = contribution()

      try {
        expect(resolve('::demo{label="hi"}')).toEqual([{ kind: 'directive', source: '::demo{label="hi"}' }])
      } finally {
        dispose()
      }
    })

    // The guarantee that makes lifting markup out of prose safe: only a name a
    // plugin is standing by to draw can be taken out of the reader's sentence.
    it('keeps an unclaimed directive as the text it always was', () => {
      expect(resolve('Pick a color. ::nobody-home{label="hi"}')).toBeNull()
    })

    it('folds an unclaimed directive into the prose beside a claimed one', () => {
      const dispose = contribution()

      try {
        expect(resolve('::nobody-home say hi ::demo{label="hi"}')).toEqual([
          { kind: 'prose', text: '::nobody-home say hi ' },
          { kind: 'directive', source: '::demo{label="hi"}' }
        ])
      } finally {
        dispose()
      }
    })
  })

  it('does not remount the widget when streaming settles', () => {
    mounts.count = 0

    const dispose = contribution({ render: MountCounter })

    try {
      const { rerender } = render(<TranscriptDirectiveLeaf streaming text="::demo" />)

      expect(mounts.count).toBe(1)
      expect(screen.getByTestId('demo-card').textContent).toBe('live')

      rerender(<TranscriptDirectiveLeaf streaming={false} text="::demo" />)

      expect(mounts.count).toBe(1)
      expect(screen.getByTestId('demo-card').textContent).toBe('settled')
    } finally {
      dispose()
    }
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
