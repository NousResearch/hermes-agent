// @vitest-environment jsdom
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { ackComposerInsert } from '@/app/chat/composer/focus'
import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { registry } from '@/contrib/registry'
import { TRANSCRIPT_DIRECTIVE_AREA, type TranscriptDirectiveContribution } from '@/lib/transcript-directives'
import { composerHost } from '@/sdk/composer'
import { $selectedStoredSessionId, setActiveSessionId } from '@/store/session'

import { TranscriptDirectiveLeaf, TranscriptDirectiveText, useResolvedParagraph } from './transcript-directive'

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

  it('renders every claimed directive in one paragraph, in order', () => {
    const dispose = contribution()

    try {
      render(<TranscriptDirectiveLeaf text='::demo{label="one"} ::demo{label="two"}' />)
      expect(screen.getAllByTestId('demo-card').map(card => card.textContent)).toEqual(['one', 'two'])
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

  it("routes a card's action to the chat that renders it, not to the active composer", async () => {
    // Primary pane shows A (and holds the active composer); a tile shows B.
    setActiveSessionId('rt-a')
    $selectedStoredSessionId.set('sess-a')

    const tileB: SessionView = {
      ...PRIMARY_SESSION_VIEW,
      $runtimeId: atom<null | string>('rt-b'),
      $storedId: atom<null | string>('sess-b'),
      kind: 'tile'
    }

    const targets: string[] = []

    const onInsert = (event: Event) => {
      const { target, token } = (event as CustomEvent<{ target: string; token: number }>).detail

      targets.push(target)
      ackComposerInsert(token, true)
    }

    // What a plugin card does: its own chat when the host names it, else the
    // active composer (the pre-`sessionId` call, unchanged).
    const dispose = contribution({
      render: props => (
        <button
          data-testid={`next-${props.attrs.label}`}
          onClick={() => void composerHost.insertText(props.sessionId ?? null, 'next step')}
          type="button"
        />
      )
    })

    window.addEventListener('hermes:composer-insert', onInsert)

    try {
      render(
        <>
          <TranscriptDirectiveLeaf text='::demo{label="a"}' />
          <SessionViewProvider value={tileB}>
            <TranscriptDirectiveLeaf text='::demo{label="b"}' />
          </SessionViewProvider>
        </>
      )

      fireEvent.click(screen.getByTestId('next-b'))
      fireEvent.click(screen.getByTestId('next-a'))
      await waitFor(() => expect(targets).toHaveLength(2))

      expect(targets).toEqual(['tile:sess-b', 'main'])
    } finally {
      window.removeEventListener('hermes:composer-insert', onInsert)
      dispose()
      setActiveSessionId(null)
      $selectedStoredSessionId.set(null)
    }
  })
})

describe('useResolvedParagraph', () => {
  const directive = (source: string) => <TranscriptDirectiveText key={source} source={source} />

  const withInline = (run: () => void) => {
    const dispose = registry.register({
      id: 'test:inline',
      area: TRANSCRIPT_DIRECTIVE_AREA,
      source: 'plugin:test',
      data: { name: 'inline', render: () => null } satisfies TranscriptDirectiveContribution
    })

    try {
      run()
    } finally {
      dispose()
    }
  }

  it('lifts a claimed directive out of the sentence around it, keeping unclaimed ones as prose', () => {
    withInline(() => {
      const other = directive('::other')

      const { result } = renderHook(() =>
        useResolvedParagraph(['See ', directive('::inline{x="1"}'), ' and ', other, ' here'])
      )

      expect(result.current?.map(segment => segment.kind)).toEqual(['prose', 'directive', 'prose'])
      expect(result.current?.[1]).toEqual({ kind: 'directive', source: '::inline{x="1"}' })
      expect(result.current?.[0]).toEqual({ kind: 'prose', children: ['See'] })
      expect(result.current?.[2]).toMatchObject({ kind: 'prose', children: ['and ', expect.anything(), ' here'] })
    })
  })

  it('keeps line breaks and inline markup in the prose, dropping only the seams around a card', () => {
    withInline(() => {
      const bold = <b key="b">bold</b>

      const { result } = renderHook(() =>
        useResolvedParagraph([
          directive('::inline{x="1"}'),
          <br key="1" />,
          directive('::inline{x="2"}'),
          <br key="2" />,
          'Some ',
          bold,
          <br key="3" />,
          'text'
        ])
      )

      expect(result.current?.map(segment => segment.kind)).toEqual(['directive', 'directive', 'prose'])

      const prose = result.current?.[2]

      expect(prose?.kind === 'prose' && prose.children.length).toBe(4)
    })
  })

  it('is null without a claimed directive, or when disabled', () => {
    withInline(() => {
      expect(renderHook(() => useResolvedParagraph(['plain ', directive('::other')])).result.current).toBeNull()
      expect(renderHook(() => useResolvedParagraph([directive('::inline')], false)).result.current).toBeNull()
    })
  })
})
