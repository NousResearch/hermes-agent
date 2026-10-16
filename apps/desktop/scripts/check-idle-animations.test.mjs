import assert from 'node:assert/strict'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { test } from 'vitest'

import { collectCss, findUncoveredAnimations } from './check-idle-animations.mjs'

const PAUSE = `:root[data-renderer-animations-paused] :is(.spinner, [data-slot='bar'], [class*='animate-spin'] svg),
:root[data-renderer-animations-paused] .arc::before { animation-play-state: paused !important; }`

// Stand-in for Tailwind's theme: infinite utilities are derived from these,
// including one that is infinite only through a custom property declared
// after it (so a single pass in source order would miss it).
const THEME = `@theme default {
  --animate-spin: spin 1s linear infinite;
  --animate-fade: fade 1s ease-out;
  --animate-wobble: var(--loop);
  --loop: wobble 2s infinite;
}`

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
  // spelling of that pseudo (with a -webkit- animation), a Tailwind @utility
  // class, and a reduced-motion block whose only branch requires `reduce`.
  assert.deepEqual(
    scan(`
      .spinner { animation: spin 1s linear infinite; }
      .spinner[data-state='on'] { animation-iteration-count: infinite; }
      .rail .tick [data-slot='bar'] { animation: load 700ms infinite alternate; }
      .host { &.arc::before { animation: sweep 2s linear infinite; } }
      .arc:before { -webkit-animation: sweep 2s linear infinite; }
      @utility spinner { animation: spin 1s infinite; }
      .once { animation: pop 200ms both; }
      @media (prefers-reduced-motion: reduce) { .loose { animation: x 1s infinite; } }
    `),
    []
  )

  // Uncovered: a new class, an ancestor-only match, a pseudo-element whose
  // base is paused but whose pseudo is not, values infinite only through
  // custom properties (one and two hops), one nested in an at-rule inside a
  // rule, one gated on no-preference, a media list with a non-reduce branch,
  // a declaration after an escaped backslash in a string, @apply with and
  // without a variant, a -webkit- animation, and a functional utility.
  assert.deepEqual(
    scan(`
      .glow { animation: pulse 2s ease-in-out infinite; }
      .spinner .inner { animation: spin 1s infinite; }
      .spinner::after { animation: spin 1s infinite; }
      .via-var { --loop2: spin 1s infinite; animation: var(--loop2); }
      .two-hops { animation: var(--animate-wobble); }
      .nested { @media (min-width: 1px) { animation: spin 1s infinite; } }
      @media (prefers-reduced-motion: no-preference) { .motion { animation: spin 1s infinite; } }
      @media screen, (prefers-reduced-motion: reduce) { .listed { animation: spin 1s infinite; } }
      @media (prefers-reduced-motion: reduce) or (min-width: 0) { .or-query { animation: spin 1s infinite; } }
      .quote::before { content: "\\\\"; }
      .after-quote { animation: spin 1s infinite; }
      .applied { @apply animate-spin; }
      .applied-variant { @apply hover:animate-wobble; }
      .spinner { @apply [&_svg]:animate-spin; }
      .webkit { -webkit-animation: spin 1s infinite; }
      @utility spinner-* { animation: spin 1s infinite; }
      /* .commented { animation: nope 1s infinite; } */
    `),
    [
      '.glow',
      '.spinner .inner',
      '.spinner::after',
      '.via-var',
      '.two-hops',
      '.nested',
      '.motion',
      '.listed',
      '.or-query',
      '.after-quote',
      '.applied',
      '.applied-variant',
      '.spinner svg',
      '.webkit',
      '.spinner-*'
    ]
  )
})

test('only an unconditional rule whose every play-state entry is paused counts as a pause', () => {
  const css = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x']
    .map(name => `.${name} { animation: spin 1s infinite; }`)
    .join('\n')

  const pause = `
    :root[data-renderer-animations-paused] .a { will-change: auto; }
    @media (prefers-reduced-motion: reduce) { :root[data-renderer-animations-paused] .b { animation-play-state: paused; } }
    @container (min-width: 1px) { :root[data-renderer-animations-paused] .c { animation-play-state: paused; } }
    :root[data-renderer-animations-paused] :is(.x, .y) > :is(.d) { animation-play-state: paused; }
    :root[data-renderer-animations-paused] .e { animation-play-state: running, paused; }
    :root[data-renderer-animations-paused].f { animation-play-state: paused; }
    @layer base { :root[data-renderer-animations-paused] { .g { animation-play-state: paused, paused; } } }
    :root[data-renderer-animations-paused] .h { -webkit-animation-play-state: paused; }
  `

  // .g (nested, inside @layer) and .h (-webkit-) are real pauses.
  assert.deepEqual(scan(css, '', pause), ['.a', '.b', '.c', '.d', '.e', '.f', '.x'])

  // A functional utility's `-*` is not a selector a browser keeps.
  assert.deepEqual(
    scan(
      '@utility spin-* { animation: spin 1s infinite; }',
      '',
      ':root[data-renderer-animations-paused] :is(.spin-*) { animation-play-state: paused; }'
    ),
    ['.spin-*']
  )
})

test('markup cannot start an infinite animation the pause rule cannot reach', () => {
  const markup = [
    '<i className="animate-spin size-3" />',
    '<i className="!animate-spin hover:animate-spin" />',
    '<i className="animate-fade" />',
    '<i className="animate-[wobble_1s_linear_infinite] animate-(--animate-spin) animate-[fade_1s]" />',
    'style={{ animation: `twinkle ${d}s ease-in-out infinite` }}',
    'style={{ animationIterationCount: "infinite" }}',
    'style={{ animation: "var(--loop)" }}',
    '<b className="animate-wobble" />'
  ].join('\n')

  // The two arbitrary values on line 4 are one gap each, reported by line.
  assert.deepEqual(scan('', markup, '').sort(), [
    '(inline style) view.tsx:4',
    '(inline style) view.tsx:5',
    '(inline style) view.tsx:6',
    '(inline style) view.tsx:7',
    '.animate-spin',
    '.animate-wobble',
    "[class*='animate-spin']"
  ])

  // A descendant variant styles the child, so only a pause on the child
  // covers it; a variant or `!` on the element needs the class-substring form.
  assert.deepEqual(scan('', '<b className="[&_svg]:animate-spin" /><i className="!animate-spin" />'), [
    "[class*='animate-spin']"
  ])
  assert.deepEqual(scan('', '<b className="[&_svg]:animate-spin" />', ''), ["[class*='animate-spin'] svg"])

  // `animation-play-state` is not inherited: child, descendant and
  // pseudo-element variants must be paused where they animate.
  assert.deepEqual(
    scan('', '<b className="*:animate-spin **:animate-spin before:animate-spin ${a}animate-wobble" />', ''),
    ["[class*='animate-spin'] > *", "[class*='animate-spin'] *", "[class*='animate-spin']::before", '.animate-wobble']
  )
})

test('the stylesheets a src file @imports are read, and an unresolvable import fails loudly', () => {
  const root = mkdtempSync(join(tmpdir(), 'idle-css-'))
  const write = (path, text) => {
    mkdirSync(dirname(join(root, path)), { recursive: true })
    writeFileSync(join(root, path), text)
  }

  try {
    write(
      'node_modules/pkg/package.json',
      JSON.stringify({
        exports: { '.': { style: { default: './main.css' } }, './theme': './theme.css', './css/*': './dist/*' }
      })
    )
    write('node_modules/pkg/main.css', "@import './deep';")
    write('node_modules/pkg/deep.css', '.deep {}')
    write('node_modules/pkg/theme.css', '.theme {}')
    write('node_modules/pkg/dist/extra.css', '.extra {}')
    write('src/local.css', '.local {}')
    write(
      'src/app.css',
      "@import 'pkg' layer(base);\n@import 'pkg/theme';\n@import 'pkg/css/extra.css';\n@import url(./local.css);\n@import 'https://x.test/y.css';"
    )

    const files = collectCss([join(root, 'src/app.css')]).map(({ file }) => file.slice(root.length + 1))

    assert.deepEqual(files.sort(), [
      'node_modules/pkg/deep.css',
      'node_modules/pkg/dist/extra.css',
      'node_modules/pkg/main.css',
      'node_modules/pkg/theme.css',
      'src/app.css',
      'src/local.css'
    ])

    write('src/broken.css', "@import 'missing-pkg';")
    assert.throws(() => collectCss([join(root, 'src/broken.css')]), /cannot resolve @import 'missing-pkg'/)
  } finally {
    rmSync(root, { force: true, recursive: true })
  }
})
