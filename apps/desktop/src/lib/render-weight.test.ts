import { describe, expect, it } from 'vitest'

import { messageFirstPaintWeight, messagePaintWeight, messageStoreWeight, RENDER_WEIGHT_CHARS } from './render-weight'

const bigResult = (chars: number) => ({
  type: 'tool-call',
  toolName: 'skill_view',
  args: { name: 'hermes-agent' },
  result: { content: 'x'.repeat(chars) }
})

describe('messageStoreWeight', () => {
  it('charges large text and tool results by character cost, not only part count', () => {
    const text = [{ type: 'text', text: 'x'.repeat(RENDER_WEIGHT_CHARS * 3) }]

    expect(messageStoreWeight(text)).toBe(4)
    expect(messageStoreWeight([bigResult(RENDER_WEIGHT_CHARS * 100)])).toBeGreaterThanOrEqual(101)
  })

  it('prices a 51KB tool output well above a plain exchange', () => {
    const heavy = messageStoreWeight([bigResult(51_236)])
    const light = messageStoreWeight([{ type: 'text', text: 'ok' }])

    expect(heavy).toBeGreaterThan(light * 50)
  })

  it('handles circular tool payloads without recursing forever', () => {
    const result: { content: string; self?: unknown } = { content: 'ok' }
    result.self = result

    expect(messageStoreWeight([{ type: 'tool-call', result }])).toBe(2)
  })

  it('bounds a single enormous payload', () => {
    const enormous = messageStoreWeight([bigResult(RENDER_WEIGHT_CHARS * 10_000)])

    expect(enormous).toBeLessThanOrEqual(302)
  })
})

describe('messagePaintWeight', () => {
  it('prices a settled activity row as the one line it renders, not its payload', () => {
    const heavy = messagePaintWeight([bigResult(RENDER_WEIGHT_CHARS * 100)])

    // The whole point: a collapsed tool row costs the same whether it wraps
    // 200 bytes or 50KB, because the payload sits behind a closed disclosure.
    expect(heavy).toBe(messagePaintWeight([bigResult(200)]))
    expect(heavy).toBeLessThan(messageStoreWeight([bigResult(RENDER_WEIGHT_CHARS * 100)]))
  })

  it('charges a reasoning block one collapsed header', () => {
    const thought = [{ type: 'reasoning', text: 'x'.repeat(RENDER_WEIGHT_CHARS * 20) }]

    expect(messagePaintWeight(thought)).toBe(1)
  })

  it('charges rendered markdown its real character cost', () => {
    const text = [{ type: 'text', text: 'x'.repeat(RENDER_WEIGHT_CHARS * 3) }]

    expect(messagePaintWeight(text)).toBe(4)
  })

  it('charges a diff by size — FileDiffPanel really does mount a row per line', () => {
    const diff = Array.from({ length: 400 }, (_, i) => `+line ${i}`).join('\n')

    const patch = messagePaintWeight([
      { type: 'tool-call', toolName: 'patch', args: { path: 'a.ts' }, result: { inline_diff: diff } }
    ])

    expect(patch).toBeGreaterThan(5)
  })

  it('prices an image card flat, however long its data URL', () => {
    const card = (chars: number) => [
      {
        type: 'tool-call',
        toolName: 'image_generate',
        args: {},
        result: { image: `data:image/png;base64,${'A'.repeat(chars)}` }
      }
    ]

    expect(messagePaintWeight(card(10_000_000))).toBe(messagePaintWeight(card(80)))
  })

  it('charges nothing for a row that renders nothing', () => {
    const hoisted = [
      {
        type: 'tool-call',
        toolName: 'todo',
        args: { todos: Array.from({ length: 40 }, (_, i) => ({ content: `t${i}` })) }
      },
      { type: 'tool-call', toolName: 'react_to_message', args: { emoji: '❤️' } }
    ]

    // Floors at 1: a message always occupies at least a row of the transcript.
    expect(messagePaintWeight(hoisted)).toBe(1)
  })

  it('keeps a tool-heavy turn far cheaper to paint than to hold', () => {
    // The measured shape behind the bad threshold: a dozen collapsed activity
    // rows and a little prose. It paints as ~a dozen lines and used to be
    // priced as an entire DOM page.
    const parts = Array.from({ length: 12 }, () => bigResult(4_000)).concat([
      { type: 'text', text: 'x'.repeat(600) } as unknown as ReturnType<typeof bigResult>
    ])

    expect(messagePaintWeight(parts)).toBeLessThan(messageStoreWeight(parts) / 5)
  })

  it('bounds a message of many enormous parts', () => {
    const parts = Array.from({ length: 50 }, () => ({
      type: 'text',
      text: 'x'.repeat(RENDER_WEIGHT_CHARS * 500)
    }))

    // One ceiling for the whole message — not one per part.
    expect(messagePaintWeight(parts)).toBeLessThanOrEqual(350)
  })
})

describe('messageFirstPaintWeight', () => {
  it('prices a lone expanded tool result by payload chars, not the collapsed line', () => {
    // A run of a single tool call mounts EXPANDED (ToolRun forces expanded at
    // count < 2), so the first-paint commit parses the whole payload while the
    // history paint weight charges one collapsed line (#127684).
    const solo = [bigResult(57_000)]

    expect(messagePaintWeight(solo)).toBe(1)
    expect(messageFirstPaintWeight(solo)).toBeGreaterThan(100)
  })

  it('agrees with the paint weight on markdown that mounts fully either way', () => {
    const text = [{ type: 'text', text: 'x'.repeat(RENDER_WEIGHT_CHARS * 3) }]

    expect(messageFirstPaintWeight(text)).toBe(messagePaintWeight(text))
  })

  it('bounds an enormous payload under the shared per-message ceiling', () => {
    const enormous = messageFirstPaintWeight([bigResult(RENDER_WEIGHT_CHARS * 10_000)])

    expect(enormous).toBeLessThanOrEqual(302)
  })

  it('floors at one unit and accepts non-array content', () => {
    expect(messageFirstPaintWeight([{ type: 'tool-call', toolName: 'todo', args: {} }])).toBeGreaterThanOrEqual(1)
    expect(messageFirstPaintWeight(null)).toBe(1)
  })
})
