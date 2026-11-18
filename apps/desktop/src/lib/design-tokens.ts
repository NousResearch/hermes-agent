/**
 * The redesign's design tokens, read from the token CSS itself (`app/redesign/tokens/*.css`), so a
 * page that lists them cannot drift from what ships. It reads the files the way
 * `scripts/check_desktop_design_tokens.py` does for DESIGN.md's tables: a comment directly before a
 * declaration is its note, the root block is the base (dark) value, `.dark` and `:not(.dark)` blocks
 * are the theme values, and anything else (an accent, a width class, a scoped surface) is an
 * override with its condition.
 */

export type TokenMode = 'base' | 'dark' | 'light'

export interface TokenOverride {
  /** What has to hold for the value to apply: an at-rule condition and/or a selector. */
  condition: string
  value: string
}

export interface DesignToken {
  name: string
  /** The source comment directly before the token's first declaration ('' when none). */
  note: string
  values: Partial<Record<TokenMode, string>>
  overrides: TokenOverride[]
}

const ROOT = ":root[data-shell='portal']"

const MODE_SELECTORS: Record<string, TokenMode> = {
  [ROOT]: 'base',
  ":root.dark[data-shell='portal']": 'dark',
  [`${ROOT}:not(.dark)`]: 'light'
}

interface Decl {
  prop: string
  value: string
  note: string
}

interface Block {
  prelude: string
  decls: Decl[]
  children: Block[]
}

const squash = (text: string) => text.replace(/\s+/g, ' ').trim()

/** Skips a quoted string starting at `start`; returns the index after its closing quote. */
function stringEnd(css: string, start: number): number {
  const quote = css[start]
  let i = start + 1

  while (i < css.length && css[i] !== quote) {
    i += css[i] === '\\' ? 2 : 1
  }

  return i + 1
}

/** Blocks with their declarations. A comment met while no declaration has started becomes the next
 *  declaration's note, as in the Python reader. */
export function parseCssBlocks(css: string): Block {
  const root: Block = { prelude: '', decls: [], children: [] }
  const stack = [root]
  let buf = ''
  let note = ''
  let depth = 0
  let i = 0

  const flush = () => {
    const statement = buf.trim()
    buf = ''

    const colon = statement.indexOf(':')

    if (colon > 0 && !statement.startsWith('@')) {
      stack
        .at(-1)!
        .decls.push({ prop: statement.slice(0, colon).trim(), value: squash(statement.slice(colon + 1)), note })
    }

    note = ''
  }

  while (i < css.length) {
    const ch = css[i]

    if (css.startsWith('/*', i)) {
      const end = css.indexOf('*/', i + 2)
      const close = end < 0 ? css.length : end

      if (!buf.trim()) {
        note = squash(css.slice(i + 2, close))
      }

      i = close + 2

      continue
    }

    if (ch === '"' || ch === "'") {
      const end = stringEnd(css, i)
      buf += css.slice(i, end)
      i = end

      continue
    }

    i += 1

    if (ch === '(' || ch === ')') {
      depth = Math.max(0, depth + (ch === '(' ? 1 : -1))
    } else if (depth === 0 && ch === '{') {
      const block: Block = { prelude: squash(buf), decls: [], children: [] }
      stack.at(-1)!.children.push(block)
      stack.push(block)
      buf = ''
      note = ''

      continue
    } else if (depth === 0 && ch === ';') {
      flush()

      continue
    } else if (depth === 0 && ch === '}') {
      flush()

      if (stack.length > 1) {
        stack.pop()
      }

      continue
    }

    buf += ch
  }

  return root
}

/** Where a root-level selector applies when it is not one of the three theme blocks: the root under an
 *  extra condition (an accent) or a scoped element below it. */
function selectorCondition(prelude: string): string {
  return prelude.startsWith(ROOT) ? prelude.slice(ROOT.length).trim() : prelude
}

function customProps(decls: Decl[]): Decl[] {
  return decls.filter(decl => decl.prop.startsWith('--'))
}

/**
 * A token file's custom properties in source order. Theme values come from the root-level theme
 * blocks (and from an at-rule that declares tokens itself, Tailwind's `@theme`); every other block
 * that sets a token is an override, noted with its condition.
 */
export function parseTokenCss(css: string): DesignToken[] {
  const tokens = new Map<string, DesignToken>()

  const token = (decl: Decl) => {
    const found = tokens.get(decl.prop) ?? { name: decl.prop, note: '', values: {}, overrides: [] }
    found.note ||= decl.note
    tokens.set(decl.prop, found)

    return found
  }

  const override = (decls: Decl[], condition: string) => {
    for (const decl of customProps(decls)) {
      token(decl).overrides.push({ condition, value: decl.value })
    }
  }

  for (const block of parseCssBlocks(css).children) {
    const mode = MODE_SELECTORS[block.prelude]

    if (mode) {
      for (const decl of customProps(block.decls)) {
        token(decl).values[mode] = decl.value
      }
    } else if (block.prelude.startsWith('@')) {
      // `@theme inline` declares tokens directly; `@media` wraps rules that override them.
      for (const decl of customProps(block.decls)) {
        token(decl).values.base = decl.value
      }

      for (const rule of block.children) {
        const ruleMode = MODE_SELECTORS[rule.prelude]
        const scope = ruleMode === 'base' ? '' : ruleMode ? ` (${ruleMode})` : ` ${selectorCondition(rule.prelude)}`
        override(rule.decls, `${block.prelude}${scope}`)
      }
    } else {
      override(block.decls, selectorCondition(block.prelude))
    }
  }

  return [...tokens.values()]
}

/** The files `index.css` imports, in cascade order (`@import './raw.css'` → `raw`). */
export function tokenImportOrder(indexCss: string): string[] {
  return [...indexCss.matchAll(/@import\s+['"]\.\/([\w-]+)\.css['"]/g)].map(match => match[1])
}

/** A file's leading comment: what the file says it is for. */
export function leadingComment(css: string): string {
  const match = /^\s*\/\*([\s\S]*?)\*\//.exec(css)

  return match ? squash(match[1]) : ''
}

export type SpecimenKind =
  | 'color'
  | 'duration'
  | 'easing'
  | 'font'
  | 'layer'
  | 'length'
  | 'radius'
  | 'shadow'
  | 'tracking'
  | 'type-size'
  | 'value'

/** `CSS.supports`, injectable so the classification is testable where no CSS engine is. */
export type SupportsFn = (property: string, value: string) => boolean

/**
 * How to show a token, from its name and its resolved value. Names decide where the value alone is
 * ambiguous (a radius and a gap are both lengths); the value decides the rest.
 */
export function specimenKind(name: string, value: string, supports: SupportsFn): SpecimenKind {
  const v = value.trim()

  if (/^-?\d*\.?\d+m?s$/.test(v)) {
    return 'duration'
  }

  if (/^(cubic-bezier|steps)\(|^ease|^linear/.test(v)) {
    return 'easing'
  }

  if (/-layer-/.test(name) && /^-?\d+$/.test(v)) {
    return 'layer'
  }

  if (/-font-/.test(name) && !/size|line|weight/.test(name)) {
    return 'font'
  }

  if (/shadow/.test(name) && supports('box-shadow', v)) {
    return 'shadow'
  }

  if (/radius/.test(name) && supports('border-radius', v) && !/^\d*\.?\d+$/.test(v)) {
    return 'radius'
  }

  if (/tracking|letter-spacing/.test(name) && supports('letter-spacing', v)) {
    return 'tracking'
  }

  if ((/-type-.*-size$/.test(name) || /^--text-ds-[\w-]+$/.test(name)) && !/--/.test(name.slice(2))) {
    return supports('font-size', v) ? 'type-size' : 'value'
  }

  if (supports('color', v)) {
    return 'color'
  }

  if (supports('width', v)) {
    return 'length'
  }

  return 'value'
}

/** A computed colour (`rgb()`, `rgba()` or `color(srgb …)`) as `#rrggbb`, plus `aa` when not opaque.
 *  Anything else comes back unchanged. */
export function hexColor(computed: string): string {
  const rgb = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/.exec(computed)
  const srgb = /^color\(srgb\s+([\d.e-]+)\s+([\d.e-]+)\s+([\d.e-]+)(?:\s*\/\s*([\d.]+%?))?\s*\)$/.exec(computed)

  const channels = rgb
    ? rgb.slice(1, 4).map(Number)
    : srgb
      ? srgb.slice(1, 4).map(channel => Number(channel) * 255)
      : null

  if (!channels) {
    return computed
  }

  const alphaText = (rgb ?? srgb)![4]
  const alpha = alphaText === undefined ? 1 : alphaText.endsWith('%') ? parseFloat(alphaText) / 100 : Number(alphaText)

  const byte = (n: number) =>
    Math.round(Math.min(255, Math.max(0, n)))
      .toString(16)
      .padStart(2, '0')

  return `#${channels.map(byte).join('')}${alpha < 1 ? byte(alpha * 255) : ''}`
}
