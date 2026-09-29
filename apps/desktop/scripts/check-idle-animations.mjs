#!/usr/bin/env node
// Idle-contract gate for CSS: every infinite animation the renderer can run
// must sleep while the window is hidden or minimized.
//
// `installRendererAnimationPauseState()` (src/lib/renderer-loop-pause.ts) puts
// `data-renderer-animations-paused` on :root while the main window cannot be
// seen. An infinite animation sleeps only if a rule under that attribute sets
// `animation-play-state: paused` on the animated element. That pause is
// opt-in per selector, so a new animation without a pause entry keeps burning
// a hidden window (#51927, #53902, #73082). This script fails the lint lane
// when an infinite animation has no matching pause rule. The contract it
// enforces is in apps/desktop/AGENTS.md, "Idle costs nothing".
//
//   node scripts/check-idle-animations.mjs          # exit 1 on gaps
//
// What it reads: every .css file under src/, plus the package and relative
// stylesheets those files @import (codicons, tw-shimmer, Tailwind's theme),
// transitively; and .ts/.tsx under src/ for Tailwind animate-* utilities,
// arbitrary `animate-[… infinite]` values and inline animation styles. The
// infinite utilities come from `--animate-*` theme variables, not a
// hard-coded list.
//
// Coverage rule: a pause selector P covers an animated selector S when S
// equals P, or when P is a single compound selector whose simple selectors
// all appear in S's last compound with the same pseudo-element. Only an
// unconditional rule that actually sets `animation-play-state: paused` counts
// as a pause. A miss means "add the exact selector to a pause rule".
//
// Known blind spots (also in AGENTS.md): Web Animations started from script
// (`element.animate(…, { iterations: Infinity })`), class names assembled at
// runtime from fragments, and windows that never install the pause state
// (the overlay, quick and wake windows).

import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { isMain } from './utils.mjs'

export const PAUSED_ROOT = ':root[data-renderer-animations-paused]'

const INFINITE = /\binfinite\b/i
const PAUSED = /\bpaused\b/i
const VAR_REF = /var\(\s*(--[\w-]+)/g
const REDUCED_MOTION = /prefers-reduced-motion\s*:\s*reduce/i
const LEGACY_PSEUDO_ELEMENT = /(^|[^:]):(before|after|first-line|first-letter)\b/g

// A class-token edge in markup: whitespace, quotes, backticks, a variant's
// `:` (`hover:animate-spin`) and Tailwind's `!` important marker on either
// side (`!animate-spin`, `animate-spin!`).
const TOKEN_EDGE = String.raw`[\s"'\`:!]`

const INLINE_PATTERNS = [
  /\banimation\s*:\s*[`'"][^`'"]*\binfinite\b/gi,
  /\banimationIterationCount\s*:\s*[`'"]infinite[`'"]/gi,
  /\banimate-\[[^\]]*infinite[^\]]*\]/gi
]

// ── CSS reader ──────────────────────────────────────────────────────────────
// Comments stripped, strings (with escapes) and parens respected, nested
// blocks kept. Returns rule / at-rule / statement (`@import …;`) /
// declaration nodes with parent links and 1-based lines. Enough to resolve
// nesting and read animation values; it is not a validating parser.
export function parseCss(text) {
  const root = { children: [], parent: null, type: 'root' }
  let current = root
  let buffer = ''
  let bufferLine = 1
  let line = 1
  let depth = 0
  let quote = null

  const atRuleName = prelude => prelude.slice(1).split(/[\s(]/)[0].toLowerCase()
  const atRuleParams = prelude => prelude.replace(/^@\S+\s*/, '')

  const flush = () => {
    const raw = buffer.trim()
    buffer = ''

    if (!raw) return

    if (raw.startsWith('@')) {
      current.children.push({
        line: bufferLine,
        name: atRuleName(raw),
        params: atRuleParams(raw),
        parent: current,
        type: 'statement'
      })

      return
    }

    const colon = raw.indexOf(':')

    if (current.type === 'root' || colon <= 0) return

    current.children.push({
      line: bufferLine,
      parent: current,
      prop: raw.slice(0, colon).trim().toLowerCase(),
      type: 'decl',
      value: raw.slice(colon + 1).trim()
    })
  }

  for (let i = 0; i < text.length; i++) {
    const ch = text[i]

    if (!quote && ch === '/' && text[i + 1] === '*') {
      const close = text.indexOf('*/', i + 2)
      const stop = close === -1 ? text.length : close + 2
      line += (text.slice(i, stop).match(/\n/g) ?? []).length
      i = stop - 1

      continue
    }

    if (ch === '\n') line++
    if (!buffer.trim() && !/\s/.test(ch)) bufferLine = line

    if (quote) {
      buffer += ch

      if (ch === '\\' && i + 1 < text.length) {
        buffer += text[++i]
        if (text[i] === '\n') line++
      } else if (ch === quote) {
        quote = null
      }

      continue
    }

    if (ch === '"' || ch === "'") quote = ch
    else if (ch === '(') depth++
    else if (ch === ')') depth--
    else if (depth === 0 && ch === '{') {
      const prelude = buffer.trim()
      buffer = ''

      // Tailwind's `@utility name { … }` defines the class `.name`; read it as
      // that rule so its declarations resolve like any other. A functional
      // utility (`name-*`) keeps the literal `-*`, which no pause selector
      // matches, so an infinite animation inside one is reported, not missed.
      const utility = prelude.match(/^@utility\s+([\w-]+(?:-\*)?)$/)?.[1]
      const node = utility
        ? { children: [], line: bufferLine, parent: current, selector: `.${utility}`, type: 'rule' }
        : prelude.startsWith('@')
          ? {
              children: [],
              line: bufferLine,
              name: atRuleName(prelude),
              params: atRuleParams(prelude),
              parent: current,
              type: 'atrule'
            }
          : { children: [], line: bufferLine, parent: current, selector: prelude, type: 'rule' }

      current.children.push(node)
      current = node

      continue
    } else if (depth === 0 && ch === '}') {
      flush()
      current = current.parent ?? root

      continue
    } else if (depth === 0 && ch === ';') {
      flush()

      continue
    }

    buffer += ch
  }

  flush()

  return root
}

function walk(node, visit) {
  for (const child of node.children ?? []) {
    visit(child)
    walk(child, visit)
  }
}

function splitTopLevel(text, sep = ',') {
  const out = []
  let depth = 0
  let quote = null
  let start = 0

  for (let i = 0; i < text.length; i++) {
    const ch = text[i]

    if (quote) {
      if (ch === '\\') i++
      else if (ch === quote) quote = null
    } else if (ch === '"' || ch === "'") {
      quote = ch
    } else if (ch === '(' || ch === '[') {
      depth++
    } else if (ch === ')' || ch === ']') {
      depth--
    } else if (ch === sep && depth === 0) {
      out.push(text.slice(start, i))
      start = i + 1
    }
  }

  out.push(text.slice(start))

  return out.map(s => s.trim()).filter(Boolean)
}

const normalize = selector =>
  selector
    .replace(/\s+/g, ' ')
    .replace(/\s*([>+~])\s*/g, ' $1 ')
    .replace(LEGACY_PSEUDO_ELEMENT, '$1::$2')
    .trim()

function ancestors(node) {
  const out = []

  for (let n = node.parent; n && n.type !== 'root'; n = n.parent) out.push(n)

  return out
}

/** Resolve nested rules to flat selectors (`&` substitution, else descendant). */
function resolvedSelectors(rule) {
  return [rule, ...ancestors(rule)]
    .filter(node => node.type === 'rule')
    .reverse()
    .map(node => splitTopLevel(node.selector))
    .reduce(
      (parents, children) =>
        parents.flatMap(parent =>
          children.map(child => {
            if (!parent) return child
            if (child.includes('&')) return child.replaceAll('&', parent)

            return `${parent} ${child}`
          })
        ),
      ['']
    )
    .map(normalize)
}

const nearestRule = node => ancestors(node).find(n => n.type === 'rule')
const inAtRule = (node, names) => ancestors(node).some(n => n.type === 'atrule' && names.includes(n.name))

const exemptByReducedMotion = node =>
  ancestors(node).some(
    n => n.type === 'atrule' && n.name === 'media' && REDUCED_MOTION.test(n.params) && !/\bnot\b/i.test(n.params)
  )

/** Index of the paren that closes the one at `open`, or -1. */
function closingParen(text, open) {
  let depth = 0

  for (let i = open; i < text.length; i++) {
    if (text[i] === '(') depth++
    else if (text[i] === ')' && --depth === 0) return i
  }

  return -1
}

/** `:root[…paused] :is(a, b) x` → [a x, b x]; `:root[…paused] a` → [a]. */
function pausedTargets(selector) {
  if (!selector.startsWith(PAUSED_ROOT)) return []

  const rest = selector.slice(PAUSED_ROOT.length).trim()

  if (rest.startsWith(':is(')) {
    const close = closingParen(rest, 3)

    if (close === -1) return []

    const tail = rest.slice(close + 1)

    return splitTopLevel(rest.slice(4, close)).map(inner => normalize(`${inner}${tail}`))
  }

  return rest ? [normalize(rest)] : []
}

// Split a compound selector into its simple selectors plus its pseudo-element,
// wherever the pseudo-element sits (`.a::before:hover` → `::before`).
function compoundParts(compound) {
  const pseudoElement = compound.match(/::[\w-]+(\([^)]*\))?/)?.[0] ?? ''
  const base = compound.replace(pseudoElement, '')
  const simples = base.match(/(\.[\w-]+|#[\w-]+|\[[^\]]+\]|:[\w-]+(\([^)]*\))?|^[a-z][\w-]*|\*)/gi) ?? []

  return { pseudoElement, simples: new Set(simples) }
}

const lastCompound = selector => splitTopLevel(selector.replace(/\s*([>+~])\s*/g, ' '), ' ').at(-1) ?? ''

export function covers(pause, animated) {
  if (pause === animated) return true
  if (/[\s>+~]/.test(pause)) return false

  const p = compoundParts(pause)
  const a = compoundParts(lastCompound(animated))

  return p.pseudoElement === a.pseudoElement && p.simples.size > 0 && [...p.simples].every(s => a.simples.has(s))
}

// ── Stylesheets reached through @import ─────────────────────────────────────

function packageEntry(pkgDir) {
  const manifest = JSON.parse(readFileSync(join(pkgDir, 'package.json'), 'utf8'))
  const dot = manifest.exports?.['.'] ?? manifest.exports
  const entry = typeof dot === 'string' ? dot : (dot?.style ?? manifest.style)

  return entry ? join(pkgDir, entry) : null
}

/** Resolve an @import target from `fromFile`; null for remote URLs and packages with no stylesheet entry. */
export function resolveCssImport(spec, fromFile) {
  if (/^(https?:|data:)/.test(spec)) return null
  if (spec.startsWith('.') || spec.startsWith('/')) return resolve(dirname(fromFile), spec)

  const scoped = spec.startsWith('@')
  const parts = spec.split('/')
  const pkgName = parts.slice(0, scoped ? 2 : 1).join('/')
  const subpath = parts.slice(scoped ? 2 : 1).join('/')

  for (let dir = dirname(resolve(fromFile)); ; dir = dirname(dir)) {
    const pkgDir = join(dir, 'node_modules', pkgName)

    if (existsSync(join(pkgDir, 'package.json'))) return subpath ? join(pkgDir, subpath) : packageEntry(pkgDir)
    if (dirname(dir) === dir) return null
  }
}

const importSpec = params => params.match(/^(?:url\()?\s*['"]([^'"]+)['"]/)?.[1] ?? null

/** The given stylesheets plus every stylesheet they @import, transitively. */
export function collectCss(entries) {
  const out = []
  const seen = new Set()
  const queue = [...entries]

  while (queue.length) {
    const file = queue.shift()

    if (seen.has(file)) continue
    seen.add(file)

    const text = readFileSync(file, 'utf8')
    out.push({ file, text })

    walk(parseCss(text), node => {
      if (node.type !== 'statement' || node.name !== 'import') return

      const spec = importSpec(node.params)
      const target = spec && resolveCssImport(spec, file)

      if (!target || !target.endsWith('.css')) return
      // A stylesheet we cannot read is a gap we cannot see: fail loudly.
      if (!existsSync(target)) throw new Error(`${file}: @import '${spec}' resolved to ${target}, which does not exist`)

      queue.push(target)
    })
  }

  return out
}

// ── The check ───────────────────────────────────────────────────────────────

/**
 * @param {{ css: {file: string, text: string}[], markup: {file: string, text: string}[] }} sources
 * @returns {{ file: string, line: number, selector: string, animation: string }[]} uncovered animations
 */
export function findUncoveredAnimations({ css, markup }) {
  const parsed = css.map(({ file, text }) => ({ file, root: parseCss(text) }))

  // Pass 1: custom properties whose value is infinite, e.g. Tailwind's
  // `--animate-spin: spin 1s linear infinite` or a local `--x: … infinite`.
  const infiniteVars = new Set()

  for (const { root } of parsed) {
    walk(root, node => {
      if (node.type === 'decl' && node.prop.startsWith('--') && INFINITE.test(node.value)) infiniteVars.add(node.prop)
    })
  }

  const isInfinite = value =>
    INFINITE.test(value) || [...value.matchAll(VAR_REF)].some(([, name]) => infiniteVars.has(name))
  const infiniteUtilities = [...infiniteVars].filter(v => v.startsWith('--animate-')).map(v => v.slice(2))
  const applyPattern = infiniteUtilities.length
    ? new RegExp(`(^|[\\s!])(${infiniteUtilities.join('|')})(?=$|[\\s!])`)
    : null

  // Pass 2: unconditional pause targets, and every infinite animation.
  const pauses = new Set()
  const animated = []

  for (const { file, root } of parsed) {
    walk(root, node => {
      if (node.type === 'rule') {
        const pausesAnimations = node.children.some(
          c => c.type === 'decl' && c.prop === 'animation-play-state' && PAUSED.test(c.value)
        )

        if (pausesAnimations && !inAtRule(node, ['media', 'supports'])) {
          for (const selector of resolvedSelectors(node)) {
            for (const target of pausedTargets(selector)) pauses.add(target)
          }
        }

        return
      }

      const rule = nearestRule(node)

      if (!rule || inAtRule(node, ['keyframes']) || exemptByReducedMotion(node)) return

      let animation = null

      if (node.type === 'statement' && node.name === 'apply')
        animation = node.params.match(applyPattern ?? /$^/)?.[2] ?? null
      else if (node.type === 'decl' && (node.prop === 'animation' || node.prop === 'animation-iteration-count'))
        animation = isInfinite(node.value) ? node.value : null

      if (!animation) return

      for (const selector of resolvedSelectors(rule)) animated.push({ animation, file, line: node.line, selector })
    })
  }

  // Markup: Tailwind utilities, arbitrary infinite animations, inline styles.
  const utilityPattern = infiniteUtilities.length
    ? new RegExp(`(^|${TOKEN_EDGE})(${infiniteUtilities.join('|')})(?=$|${TOKEN_EDGE})`, 'g')
    : null

  for (const { file, text } of markup) {
    const hasUtility = text.includes('animate-')

    if (!hasUtility && !INFINITE.test(text)) continue

    const lineAt = offset => text.slice(0, offset).split('\n').length

    for (const pattern of INLINE_PATTERNS) {
      for (const match of text.matchAll(pattern)) {
        const line = lineAt(match.index)

        animated.push({ animation: match[0], file, line, selector: `(inline style) ${file}:${line}` })
      }
    }

    if (utilityPattern && hasUtility) {
      for (const match of text.matchAll(utilityPattern)) {
        animated.push({ animation: match[2], file, line: lineAt(match.index), selector: `.${match[2]}` })
      }
    }
  }

  const pauseList = [...pauses]
  const seen = new Set()

  return animated.filter(entry => {
    if (pauseList.some(pause => covers(pause, entry.selector))) return false
    // One report per selector: a utility used in 100 files is one gap.
    if (seen.has(entry.selector)) return false
    seen.add(entry.selector)

    return true
  })
}

function listFiles(dir, extensions) {
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter(entry => entry.isFile() && extensions.some(ext => entry.name.endsWith(ext)))
    .map(entry => join(entry.parentPath ?? entry.path, entry.name))
    .filter(file => !/\.(test|spec)\.[jt]sx?$/.test(file))
}

function main() {
  const desktopRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
  const src = join(desktopRoot, 'src')
  const rel = file => relative(desktopRoot, file)

  const uncovered = findUncoveredAnimations({
    css: collectCss(listFiles(src, ['.css'])).map(({ file, text }) => ({ file: rel(file), text })),
    markup: listFiles(src, ['.tsx', '.ts']).map(file => ({ file: rel(file), text: readFileSync(file, 'utf8') }))
  })

  if (uncovered.length === 0) {
    console.log('check-idle-animations: every infinite animation pauses with the window')

    return
  }

  console.error(
    `check-idle-animations: ${uncovered.length} infinite animation(s) keep running while the window is hidden.\n` +
      `Add each selector to a \`${PAUSED_ROOT}\` rule that sets \`animation-play-state: paused\`\n` +
      '(the shared list lives in src/styles.css). See apps/desktop/AGENTS.md, "Idle costs nothing".\n'
  )

  for (const { animation, file, line, selector } of uncovered) {
    console.error(`  ${file}:${line}  ${selector}  { ${animation.replace(/\s+/g, ' ')} }`)
  }

  process.exitCode = 1
}

if (isMain(import.meta.url)) main()
