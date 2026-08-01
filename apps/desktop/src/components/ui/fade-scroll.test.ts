import { describe, expect, it } from 'vitest'

import { edgeMask, scrollEdges } from './fade-scroll'

const box = (scrollTop: number, clientHeight: number, scrollHeight: number) => ({
  clientHeight,
  scrollHeight,
  scrollTop
})

describe('scrollEdges', () => {
  it('reports no clipped edge when the content fits', () => {
    expect(scrollEdges(box(0, 150, 150))).toEqual({ above: false, below: false })
  })

  it('reports only the bottom at the top of a clipped list', () => {
    expect(scrollEdges(box(0, 150, 400))).toEqual({ above: false, below: true })
  })

  it('reports both edges mid-scroll', () => {
    expect(scrollEdges(box(100, 150, 400))).toEqual({ above: true, below: true })
  })

  it('reports only the top once scrolled to the end', () => {
    expect(scrollEdges(box(250, 150, 400))).toEqual({ above: true, below: false })
  })
})

describe('edgeMask', () => {
  // The whole point of an edge-AWARE fade: content that fits is never dimmed.
  it('is absent when nothing is clipped', () => {
    expect(edgeMask({ above: false, below: false })).toBeUndefined()
  })

  it('leaves the first row fully opaque when nothing is hidden above it', () => {
    expect(edgeMask({ above: false, below: true })).toBe(
      'linear-gradient(to bottom, black, black calc(100% - 1.25rem), transparent)'
    )
  })

  it('leaves the last row fully opaque once the bottom is reached', () => {
    expect(edgeMask({ above: true, below: false })).toBe('linear-gradient(to bottom, transparent, black 1.25rem, black)')
  })

  it('fades both edges mid-scroll', () => {
    expect(edgeMask({ above: true, below: true })).toBe(
      'linear-gradient(to bottom, transparent, black 1.25rem, black calc(100% - 1.25rem), transparent)'
    )
  })
})
