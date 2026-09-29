import assert from 'node:assert/strict'
import { test } from 'vitest'

import { findUncoveredAnimations } from './check-idle-animations.mjs'

const PAUSE = `:root[data-renderer-animations-paused] :is(.spinner, [data-slot='bar']),
:root[data-renderer-animations-paused] .arc::before { animation-play-state: paused !important; }`

const scan = (css, markup = '') =>
  findUncoveredAnimations({
    css: [
      { file: 'pause.css', text: PAUSE },
      { file: 'feature.css', text: css }
    ],
    markup: markup ? [{ file: 'view.tsx', text: markup }] : []
  }).map(entry => entry.selector)

test('an infinite animation passes only when a pause rule reaches the element it animates', () => {
  // Covered: exact class, a qualified form of it, an attribute target under an
  // ancestor, and a pseudo-element listed with its pseudo.
  assert.deepEqual(
    scan(`
      .spinner { animation: spin 1s linear infinite; }
      .spinner[data-state='on'] { animation-iteration-count: infinite; }
      .rail .tick [data-slot='bar'] { animation: load 700ms infinite alternate; }
      .host { &.arc::before { animation: sweep 2s linear infinite; } }
      .once { animation: pop 200ms both; }
      @media (prefers-reduced-motion: reduce) { .loose { animation: x 1s infinite; } }
    `),
    []
  )

  // Uncovered: a new class, an ancestor-only match (the pause hits the
  // parent, not the animated child), and a pseudo-element whose base is
  // paused but whose pseudo is not.
  assert.deepEqual(
    scan(`
      .glow { animation: pulse 2s ease-in-out infinite; }
      .spinner .inner { animation: spin 1s infinite; }
      .spinner::after { animation: spin 1s infinite; }
      /* .commented { animation: nope 1s infinite; } */
    `),
    ['.glow', '.spinner .inner', '.spinner::after']
  )
})

test('markup cannot start an infinite animation the pause rule cannot reach', () => {
  assert.deepEqual(
    scan(
      '',
      ['<i className="animate-spin size-3" />', 'style={{ animation: `twinkle ${d}s ease-in-out infinite` }}'].join(
        '\n'
      )
    ),
    ['.animate-spin', '(inline style)']
  )
})
