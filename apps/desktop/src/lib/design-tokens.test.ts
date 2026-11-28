import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { hexColor, leadingComment, parseTokenCss, specimenKind, tokenImportOrder } from './design-tokens'

// The shipped token files' text (vitest hands CSS imports over empty, `?raw` included).
const TOKENS = join(dirname(fileURLToPath(import.meta.url)), '../app/redesign/tokens')

const tokenFiles = readdirSync(TOKENS)
  .filter(name => name.endsWith('.css'))
  .map(name => ({ name, css: readFileSync(join(TOKENS, name), 'utf8') }))

/** Every custom property a file declares anywhere, found without the parser: comments blanked, then
 *  each `--name:` that starts a declaration. */
function declaredNames(css: string): Set<string> {
  const code = css.replace(/\/\*[\s\S]*?\*\//g, '')

  return new Set([...code.matchAll(/(?:^|[;{])\s*(--[\w-]+)\s*:/gm)].map(match => match[1]))
}

describe('parseTokenCss on the shipped token files', () => {
  it.each(tokenFiles)('$name: every declared custom property is a token, and nothing else is', ({ css }) => {
    expect(new Set(parseTokenCss(css).map(token => token.name))).toEqual(declaredNames(css))
  })

  it('reads theme values and overrides from every kind of block the folder uses', () => {
    const tokens = tokenFiles.flatMap(file => parseTokenCss(file.css))

    // Light (`:not(.dark)`) and dark (`.dark`) blocks, the accent blocks and the width-class media.
    expect(tokens.some(token => token.values.light !== undefined)).toBe(true)
    expect(tokens.some(token => token.values.dark !== undefined)).toBe(true)
    expect(tokens.some(token => token.overrides.some(o => o.condition.startsWith('[data-portal-accent=')))).toBe(true)
    expect(tokens.some(token => token.overrides.some(o => o.condition.startsWith('@media (max-width')))).toBe(true)
  })

  it('imports every token file from index.css except the Tailwind map, which styles.css imports', () => {
    const imported = tokenImportOrder(tokenFiles.find(file => file.name === 'index.css')!.css)
    const files = tokenFiles.map(file => file.name.replace(/\.css$/, '')).filter(stem => stem !== 'index')

    expect(new Set([...imported, 'tailwind'])).toEqual(new Set(files))
  })
})

describe('parseTokenCss', () => {
  const css = `/* The file. */
:root[data-shell='portal'] {
  /* A note; with a colon. */
  --portal-a: color-mix(in srgb, var(--x) 5%, transparent);
  --portal-b: 'Oswald', serif;
  --portal-c: url('data:a;b');
}

:root[data-shell='portal']:not(.dark) {
  --portal-a: #fff;
}

:root.dark[data-shell='portal'] {
  --portal-b: mono;
}

:root[data-shell='portal'][data-portal-accent='pink'] {
  --portal-c: red;
}

@media (max-width: 639.98px) {
  :root[data-shell='portal'] {
    --portal-a: 1rem;
  }
}

@theme inline {
  --text-ds-ui: var(--portal-a);
}

@font-face {
  font-family: 'Oswald';
}`

  const tokens = Object.fromEntries(parseTokenCss(css).map(token => [token.name, token]))

  it('keeps notes, theme values and overrides apart', () => {
    expect(tokens['--portal-a']).toEqual({
      name: '--portal-a',
      note: 'A note; with a colon.',
      values: { base: 'color-mix(in srgb, var(--x) 5%, transparent)', light: '#fff' },
      overrides: [{ condition: '@media (max-width: 639.98px)', value: '1rem' }]
    })
    expect(tokens['--portal-b'].values).toEqual({ base: "'Oswald', serif", dark: 'mono' })
    expect(tokens['--portal-c'].values.base).toBe("url('data:a;b')")
    expect(tokens['--portal-c'].overrides).toEqual([{ condition: "[data-portal-accent='pink']", value: 'red' }])
    expect(tokens['--text-ds-ui'].values.base).toBe('var(--portal-a)')
    expect(Object.keys(tokens)).toHaveLength(4)
  })

  it('reads the leading comment as what the file is for', () => {
    expect(leadingComment(css)).toBe('The file.')
  })
})

describe('specimenKind', () => {
  // A stand-in for CSS.supports that knows colours and lengths only.
  const supports = (property: string, value: string) =>
    property === 'color'
      ? /^(#|rgb|color-mix|transparent)/.test(value)
      : /^-?[\d.]+(px|rem|em|%)$/.test(value) || (property === 'box-shadow' && value.includes('px'))

  it.each([
    ['--portal-motion-fast', '120ms', 'duration'],
    ['--portal-ease', 'cubic-bezier(0.2, 0.8, 0.2, 1)', 'easing'],
    ['--portal-layer-sheet', '45', 'layer'],
    ['--portal-font-display', "'Oswald', sans-serif", 'font'],
    ['--portal-shadow-floating', '0 8px 24px #000', 'shadow'],
    ['--portal-radius-card', '4px', 'radius'],
    ['--radius-scalar', '0.6', 'value'],
    ['--portal-tracking-chip', '0.04em', 'tracking'],
    ['--portal-type-body-size', '0.875rem', 'type-size'],
    ['--text-ds-caption', '0.625rem', 'type-size'],
    ['--text-ds-caption--line-height', '1.4', 'value'],
    ['--portal-surface-card', 'color-mix(in srgb, #fff 2%, transparent)', 'color'],
    ['--portal-page-top', '1.125rem', 'length'],
    ['--portal-raw-navy-channels', '11 11 36', 'value']
  ])('%s: %s is a %s', (name, value, kind) => {
    expect(specimenKind(name, value, supports)).toBe(kind)
  })
})

describe('hexColor', () => {
  it.each([
    ['rgb(0, 0, 242)', '#0000f2'],
    ['rgba(255, 255, 255, 0.5)', '#ffffff80'],
    ['color(srgb 1 1 1 / 0.05)', '#ffffff0d'],
    ['color(srgb 0.5 0 1)', '#8000ff'],
    ['none', 'none']
  ])('%s → %s', (computed, hex) => {
    expect(hexColor(computed)).toBe(hex)
  })
})
