import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

/**
 * What the reader actually sees while an answer streams, frame by frame.
 *
 * The desktop renders `preprocessMarkdown → tailBoundedRemend → Streamdown` on
 * the accumulated text every flush, so a defect in any of those steps shows up
 * here as characters missing from the DOM on some frame — and only here: a
 * settled message is preprocessed once, so "streaming is broken, static text is
 * fine" is the shape of every bug in this file.
 */
const norm = (value: string) => value.replace(/\s+/g, ' ').trim()

afterEach(cleanup)

/** Stream `text` into the real surface in `chunk`-character deltas, calling
 *  `onFrame` with the DOM text after each flush. */
async function streamFrames(text: string, chunk: number, onFrame: (visible: string) => void) {
  const { container, rerender } = render(<MarkdownTextContent isRunning text="" />)

  try {
    for (let end = chunk; end < text.length; end += chunk) {
      const frame = text.slice(0, end)

      await act(async () => {
        rerender(<MarkdownTextContent isRunning text={frame} />)
        await Promise.resolve()
      })

      onFrame(container.textContent ?? '')
    }

    await act(async () => {
      rerender(<MarkdownTextContent isRunning text={text} />)
      await Promise.resolve()
    })

    onFrame(container.textContent ?? '')
  } finally {
    cleanup()
  }
}

// Accents (BMP) plus an astral-plane emoji: a surrogate pair the pipeline must
// never split.
const ACCENTED = 'Açúcar, coração, órgão: a renderização no Hermes Desktop está correta 🎉 sem perder nada.'
// Plain long prose — the protection case: no markdown, no reasoning, no escapes.
const PLAIN =
  'Vou investigar o pipeline de renderização de texto porque o usuário relatou que palavras inteiras ' +
  'somem durante o streaming e que palavras adjacentes ficam coladas na mesma frase inteira do parágrafo.'
const REASONED = '<thinking>let me think about the render pipeline in detail</thinking>Resposta final com acentuação.'

describe('streamed markdown keeps every visible character', () => {
  it.each([1, 3])('renders accents and emoji intact when streamed in %i-character deltas', async chunk => {
    const frames: string[] = []

    await streamFrames(ACCENTED, chunk, visible => frames.push(visible))

    expect(frames.length).toBeGreaterThan(10)

    for (const visible of frames) {
      // No frame may show anything that has not arrived yet (raw, spaces and
      // all), and none may lose a word.
      expect(ACCENTED.startsWith(visible)).toBe(true)
      expect(norm(ACCENTED).includes(norm(visible))).toBe(true)
    }

    // Character-for-character, spaces included.
    expect(frames.at(-1)).toBe(ACCENTED)
  })

  it.each([1, 3])('loses nothing from plain prose streamed in %i-character deltas', async chunk => {
    const frames: string[] = []

    await streamFrames(PLAIN, chunk, visible => frames.push(visible))

    expect(frames.length).toBeGreaterThan(10)

    let previous = ''

    for (const visible of frames) {
      const current = norm(visible)

      // Every frame is a prefix of the finished text...
      expect(norm(PLAIN).startsWith(current)).toBe(true)
      // ...and nothing already on screen ever disappears.
      expect(current.startsWith(previous)).toBe(true)

      previous = current
    }

    expect(norm(frames.at(-1) ?? '')).toBe(norm(PLAIN))
  })

  it('never paints the chain of thought, in any frame', async () => {
    const frames: string[] = []

    await streamFrames(REASONED, 3, visible => frames.push(visible))

    expect(frames.length).toBeGreaterThan(10)

    for (const visible of frames) {
      expect(visible).not.toContain('let me think')
      expect(visible).not.toContain('render pipeline')
      expect(visible).not.toContain('<thinking')
      expect(visible).not.toContain('</thinking')
    }

    expect(frames.at(-1)).toBe('Resposta final com acentuação.')
  })
})
