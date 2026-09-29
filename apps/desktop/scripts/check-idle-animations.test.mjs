import assert from 'node:assert/strict'
import { test } from 'vitest'

import { findUncoveredAnimations } from './check-idle-animations.mjs'

const PAUSE = `:root[data-renderer-animations-paused] :is(.spinner, [data-slot='bar']),
:root[data-renderer-animations-paused] .arc::before { animation-play-state: paused !important; }`

// Stand-in for Tailwind's theme: infinite utilities are derived from these.
const THEME = '@theme default { --animate-spin: spin 1s linear infinite; --animate-fade: fade 1s ease-out; }'

const scan = (css, markup = '', pause = PAUSE) =>
  findUncoveredAnimations({
    css: [
      { file: 'pause.css', text: pause },
      { file: 'theme.css', text: THEME },
      { file: 'feature.css', text: css }
    ],
    markup: markup ? [{ file: 'view.tsx', text: markup }] : []
  }).map(entry => entry.selector)

test('an infinite animation passes only when a pause rule reaches the element it animates', () => {
  // Covered: exact class, a qualified form of it, an attribute target under an
  // ancestor, a pseudo-element listed with its pseudo, the legacy one-colon
  // spelling of that pseudo, and a Tailwind @utility class.
  assert.deepEqual(
    scan(`
      .spinner { animation: spin 1s linear infinite; }
      .spinner[data-state='on'] { animation-iteration-count: infinite; }
      .rail .tick [data-slot='bar'] { animation: load 700ms infinite alternate; }
      .host { &.arc::before { animation: sweep 2s linear infinite; } }
      .arc:before { animation: sweep 2s linear infinite; }
      @utility spinner { animation: spin 1s infinite; }
      .once { animation: pop 200ms both; }
      @media (prefers-reduced-motion: reduce) { .loose { animation: x 1s infinite; } }
    `),
    []
  )

  // Uncovered: a new class, an ancestor-only match, a pseudo-element whose
  // base is paused but whose pseudo is not, a value that is infinite only
  // through a custom property, one nested in an at-rule inside a rule, one
  // gated on no-preference (the usual home for motion), a declaration after
  // an escaped backslash in a string, and an @apply of an infinite utility.
  assert.deepEqual(
    scan(`
      .glow { animation: pulse 2s ease-in-out infinite; }
      .spinner .inner { animation: spin 1s infinite; }
      .spinner::after { animation: spin 1s infinite; }
      .via-var { --loop: spin 1s infinite; animation: var(--loop); }
      .nested { @media (min-width: 1px) { animation: spin 1s infinite; } }
      @media (prefers-reduced-motion: no-preference) { .motion { animation: spin 1s infinite; } }
      .quote::before { content: "\\\\"; }
      .after-quote { animation: spin 1s infinite; }
      .applied { @apply animate-spin; }
      /* .commented { animation: nope 1s infinite; } */
    `),
    ['.glow', '.spinner .inner', '.spinner::after', '.via-var', '.nested', '.motion', '.after-quote', '.applied']
  )
})

test('only an unconditional rule that sets animation-play-state: paused counts as a pause', () => {
  const css =
    '.a { animation: spin 1s infinite; } .b { animation: spin 1s infinite; } .c { animation: spin 1s infinite; } .x { animation: spin 1s infinite; }'

  const pause = `
    :root[data-renderer-animations-paused] .a { will-change: auto; }
    @media (prefers-reduced-motion: reduce) { :root[data-renderer-animations-paused] .b { animation-play-state: paused; } }
    :root[data-renderer-animations-paused] :is(.x, .y) > :is(.c) { animation-play-state: paused; }
  `

  assert.deepEqual(scan(css, '', pause), ['.a', '.b', '.c', '.x'])
})

test('markup cannot start an infinite animation the pause rule cannot reach', () => {
  const markup = [
    '<i className="animate-spin size-3" />',
    '<i className="!animate-spin" />',
    '<i className="animate-fade" />',
    '<i className="animate-[wobble_1s_linear_infinite]" />',
    'style={{ animation: `twinkle ${d}s ease-in-out infinite` }}',
    'style={{ animationIterationCount: "infinite" }}'
  ].join('\n')

  assert.deepEqual(scan('', markup, '').sort(), [
    '(inline style) view.tsx:4',
    '(inline style) view.tsx:5',
    '(inline style) view.tsx:6',
    '.animate-spin'
  ])

  // `!animate-spin` alone is still seen.
  assert.deepEqual(scan('', '<i className="!animate-spin" />', ''), ['.animate-spin'])
})
