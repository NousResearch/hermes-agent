#!/usr/bin/env node
// Idle-contract gate for CSS: every infinite animation in the renderer must
// sleep when the window is hidden or minimized.
//
// `installRendererAnimationPauseState()` (src/lib/renderer-loop-pause.ts) puts
// `data-renderer-animations-paused` on :root while the main window cannot be
// seen. An infinite animation sleeps only when a rule under that attribute
// sets `animation-play-state: paused` on the animated element. That pause is
// opt-in per selector, and a new animation without its pause entry keeps
// burning a hidden window. That shape is behind #51927, #53902 and #73082.
// This script fails the lint lane when an infinite animation has no matching
// pause rule. The apps/desktop/AGENTS.md section "Idle costs nothing" is the
// contract it enforces.
//
//   node scripts/check-idle-animations.mjs          # scan src/, exit 1 on gaps
//
// Coverage rule, deliberately conservative: a pause selector P covers an
// animated selector S when S equals P, or when P is a single compound selector
// (no combinators) whose simple selectors all appear in S's last compound with
// the same pseudo-element. A miss only means "add the exact selector to a
// pause rule". It never lets an uncovered animation through.

import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

export const PAUSED_ROOT = ':root[data-renderer-animations-paused]'

// Tailwind's built-in infinite utilities. They come from the framework, not
// from src CSS, so the scanner checks usage in markup and requires a pause
// rule for each utility that is actually used.
export const TAILWIND_INFINITE_UTILITIES = ['animate-spin', 'animate-pulse', 'animate-ping', 'animate-bounce']

const INFINITE = /\binfinite\b/
const INLINE_INFINITE = /\banimation\s*:\s*[`'"][^`'"\n]*\binfinite\b/

/**
 * Minimal CSS reader: comments stripped, strings and parens respected, nested
 * blocks kept. It returns rule / at-rule / declaration nodes with parent links
 * and 1-based line numbers. That is enough to resolve nesting and read
 * `animation` values; it is not a validating parser.
 */
export function parseCss(text) {
  const root = { children: [], parent: null, type: 'root' }
  let current = root
  let buffer = ''
  let bufferLine = 1
  let line = 1
  let depth = 0
  let quote = null

  const flushDecl = () => {
    const raw = buffer.trim()
    buffer = ''
    const colon = raw.indexOf(':')

    if (!raw || current.type === 'root' || colon <= 0) return

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
      if (ch === quote && text[i - 1] !== '\\') quote = null
      buffer += ch

      continue
    }

    if (ch === '"' || ch === "'") quote = ch
    else if (ch === '(') depth++
    else if (ch === ')') depth--
    else if (depth === 0 && ch === '{') {
      const prelude = buffer.trim()
      buffer = ''

      const node = prelude.startsWith('@')
        ? {
            children: [],
            line: bufferLine,
            name: prelude.slice(1).split(/\s/)[0],
            params: prelude.replace(/^@\S+\s*/, ''),
            parent: current,
            type: 'atrule'
          }
        : { children: [], line: bufferLine, parent: current, selector: prelude, type: 'rule' }

      current.children.push(node)
      current = node

      continue
    } else if (depth === 0 && ch === '}') {
      flushDecl()
      current = current.parent ?? root

      continue
    } else if (depth === 0 && ch === ';') {
      flushDecl()

      continue
    }

    buffer += ch
  }

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
      if (ch === quote) quote = null
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
    .trim()

/** Resolve nested rules to flat selectors (`&` substitution, else descendant). */
function resolvedSelectors(rule) {
  const chain = []

  for (let node = rule; node && node.type !== 'root'; node = node.parent) {
    if (node.type === 'rule') chain.unshift(splitTopLevel(node.selector))
  }

  return chain
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

function insideReducedMotion(node) {
  for (let n = node.parent; n; n = n.parent) {
    if (n.type === 'atrule' && n.name === 'media' && /prefers-reduced-motion/.test(n.params)) return true
  }

  return false
}

/** `:root[…paused] :is(a, b)` → [a, b]; `:root[…paused] a` → [a]. */
function pausedTargets(selector) {
  if (!selector.startsWith(PAUSED_ROOT)) return []

  const rest = selector.slice(PAUSED_ROOT.length).trim()
  const is = rest.match(/^:is\(([\s\S]*)\)(.*)$/)

  if (is) return splitTopLevel(is[1]).map(inner => normalize(`${inner}${is[2]}`))

  return rest ? [normalize(rest)] : []
}

// Split a compound selector into its simple selectors plus its pseudo-element.
function compoundParts(compound) {
  const pseudoElement = compound.match(/::[\w-]+(\([^)]*\))?$/)?.[0] ?? ''
  const base = pseudoElement ? compound.slice(0, -pseudoElement.length) : compound
  const simples = base.match(/(\.[\w-]+|#[\w-]+|\[[^\]]+\]|:{1}[\w-]+(\([^)]*\))?|^[a-z][\w-]*|\*)/gi) ?? []

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

/**
 * Scan CSS sources and markup sources.
 * @param {{ css: {file: string, text: string}[], markup: {file: string, text: string}[] }} sources
 * @returns {{ file: string, line: number, selector: string, animation: string }[]} uncovered animations
 */
export function findUncoveredAnimations({ css, markup }) {
  const pauses = new Set()
  const animated = []

  for (const { file, text } of css) {
    walk(parseCss(text), node => {
      if (node.type === 'rule') {
        for (const selector of resolvedSelectors(node)) {
          for (const target of pausedTargets(selector)) pauses.add(target)
        }

        return
      }

      if (node.type !== 'decl') return
      if (node.prop !== 'animation' && node.prop !== 'animation-iteration-count') return
      if (!INFINITE.test(node.value) || node.parent?.type !== 'rule' || insideReducedMotion(node)) return

      for (const selector of resolvedSelectors(node.parent)) {
        if (!selector.startsWith(PAUSED_ROOT)) animated.push({ animation: node.value, file, line: node.line, selector })
      }
    })
  }

  for (const { file, text } of markup) {
    text.split('\n').forEach((line, index) => {
      // An inline `animation: … infinite` style cannot be reached by a pause
      // selector at all; it has to move onto a class.
      if (INLINE_INFINITE.test(line)) {
        animated.push({
          animation: 'inline style',
          file,
          key: `${file}:${index + 1}`,
          line: index + 1,
          selector: '(inline style)'
        })
      }

      for (const utility of TAILWIND_INFINITE_UTILITIES) {
        if (new RegExp(`(^|[\\s"'\`:])${utility}($|[\\s"'\`])`).test(line)) {
          animated.push({ animation: utility, file, line: index + 1, selector: `.${utility}` })
        }
      }
    })
  }

  const pauseList = [...pauses]
  const seen = new Set()

  return animated.filter(entry => {
    if (pauseList.some(pause => covers(pause, entry.selector))) return false

    // One report per selector: a utility used in 100 files is one gap.
    const key = entry.key ?? entry.selector

    if (seen.has(key)) return false
    seen.add(key)

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
  const read = file => ({ file: relative(desktopRoot, file), text: readFileSync(file, 'utf8') })

  const uncovered = findUncoveredAnimations({
    css: listFiles(src, ['.css']).map(read),
    markup: listFiles(src, ['.tsx', '.ts']).map(read)
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
    console.error(`  ${file}:${line}  ${selector}  { ${animation} }`)
  }

  process.exitCode = 1
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main()
