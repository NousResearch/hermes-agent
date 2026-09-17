// Compare flattened translation key coverage between an English locale file
// and a translated locale file. Exits non-zero on any mismatch so CI can gate.
//
// Handles both locale formats Hermes ships:
//   1. YAML catalogs (locales/<lang>.yaml)
//   2. TS locale files (web|apps/desktop/src/i18n/<lang>.ts) built with
//      defineLocale(); function-valued leaves (interpolators) are compared
//      structurally — an en leaf that is a function requires the translated
//      leaf to be a function too, and vice versa.
//
// The TS parser is a scanner, not a compiler: it reads object literals with
// string / template / array / function leaves and tolerates TS type
// annotations on function params, e.g. `deleteTitle: (name: string) => ...`.
// No dependencies; runs on plain Node.
//
// Usage:
//   node scripts/check_i18n_parity.mjs <en-file> <translated-file> [--partial] [--verbose]
//   node scripts/check_i18n_parity.mjs --all [--verbose]
//
// --partial  tolerate missing keys (defineLocale() deep-merge locales); stale
//            keys and string<->interpolator shape drift still fail.
// --all      check every locale of every surface against its English source
//            (see SURFACES): strict for the YAML agent catalogs, partial for
//            the TS web/desktop locales.
//
// Exit codes: 0 = parity, 1 = mismatch, 2 = usage/parse error.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// ─── Shared flattening ──────────────────────────────────────────────────────

function flatten(obj, prefix = '', out = {}) {
  for (const k in obj) {
    const v = obj[k]
    const key = prefix ? `${prefix}.${k}` : k
    // Leaf markers from the TS parser ({str}/{fn}/{ref}) are leaves, not
    // containers — don't recurse into them.
    const isMarker = v && typeof v === 'object' && (v.str === true || v.fn === true || v.ref === true)
    if (v && typeof v === 'object' && !Array.isArray(v) && !isMarker) flatten(v, key, out)
    else out[key] = v
  }
  return out
}

// ─── YAML parsing (subset: nested maps, quoted/plain scalar strings) ────────

function parseYamlLocale(file) {
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
  const root = {}
  const stack = [{ indent: -1, obj: root }]
  for (let li = 0; li < lines.length; li++) {
    const raw = lines[li]
    if (!raw.trim() || /^\s*#/.test(raw)) continue
    const m = raw.match(/^(\s*)("[^"]+"|'[^']+'|[^:#]+):\s*(.*)$/)
    if (!m) continue
    const indent = m[1].length
    const key = m[2].replace(/^["']|["']$/g, '')
    let rest = m[3]
    // A quoted scalar may span physical lines (YAML folds the newlines).
    // Continuation lines must be swallowed here — a line like
    // `Use quotes around titles, for example: \`/resume \"X\"\`.` otherwise
    // parses as a bogus nested map and shifts every key under it. Key parity
    // only needs the span consumed, not the folded value.
    const q = rest[0]
    if ((q === '"' || q === "'") && !scalarClosed(rest, q)) {
      while (++li < lines.length) {
        rest += '\n' + lines[li]
        if (scalarClosed(rest, q)) break
      }
    }
    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop()
    const parent = stack[stack.length - 1].obj
    if (rest === '') {
      const child = {}
      parent[key] = child
      stack.push({ indent, obj: child })
    } else {
      parent[key] = rest
    }
  }
  return root
}

/** True when the quoted YAML scalar `s` (opening quote `q`) has its close. */
function scalarClosed(s, q) {
  for (let i = 1; i < s.length; i++) {
    if (q === '"' && s[i] === '\\') { i++; continue }
    if (s[i] === q) {
      if (q === "'" && s[i + 1] === "'") { i++; continue } // '' escape
      return true
    }
  }
  return false
}

// ─── TS parsing (bracket scanner over the exported object literal) ─────────

/**
 * Return the text of the first top-level object literal after `export`.
 * Tracks strings, template literals, line/block comments and brace depth.
 */
function extractTopObjectText(src) {
  const start = src.indexOf('export')
  if (start === -1) throw new Error('no `export` found')
  const braceStart = src.indexOf('{', start)
  if (braceStart === -1) throw new Error('no object literal after export')
  let depth = 0
  let inS = false, inD = false, inT = false, inBlock = false, inLine = false
  for (let i = braceStart; i < src.length; i++) {
    const c = src[i]
    const n = src[i + 1]
    if (inLine) { if (c === '\n') inLine = false; continue }
    if (inBlock) { if (c === '*' && n === '/') { inBlock = false; i++ } continue }
    if (inS) { if (c === '\\') { i++; continue } if (c === "'") inS = false; continue }
    if (inD) { if (c === '\\') { i++; continue } if (c === '"') inD = false; continue }
    if (inT) { if (c === '\\') { i++; continue } if (c === '`') inT = false; continue }
    if (c === '/' && n === '/') { inLine = true; i++; continue }
    if (c === '/' && n === '*') { inBlock = true; i++; continue }
    if (c === "'") { inS = true; continue }
    if (c === '"') { inD = true; continue }
    if (c === '`') { inT = true; continue }
    if (c === '{') depth++
    else if (c === '}') {
      depth--
      if (depth === 0) return src.slice(braceStart, i + 1)
    }
  }
  throw new Error('unterminated object literal')
}

/**
 * Parse an object literal (as produced by extractTopObjectText) into a tree.
 * Leaf kinds: { str } (string/template literal) or { fn } (anything else —
 * function, identifier, spread). Arrays are preserved as JS arrays of leaves.
 */
function parseObjectLiteral(text) {
  let i = 0
  const len = text.length
  const ws = () => {
    while (i < len) {
      const c = text[i]
      if (/\s/.test(c)) { i++; continue }
      if (c === '/' && text[i + 1] === '/') { while (i < len && text[i] !== '\n') i++; continue }
      if (c === '/' && text[i + 1] === '*') { i += 2; while (i < len && !(text[i] === '*' && text[i + 1] === '/')) i++; i += 2; continue }
      break
    }
  }
  const skipString = (q) => {
    let tplDepth = 0
    i++ // opening quote
    while (i < len) {
      const c = text[i]
      if (c === '\\') { i += 2; continue }
      if (q === '`' && c === '$' && text[i + 1] === '{') { tplDepth++; i += 2; continue }
      if (q === '`' && c === '}' && tplDepth > 0) { tplDepth--; i++; continue }
      if (c === q) { i++; return }
      i++
    }
    throw new Error('unterminated string')
  }

  /** Consume an arrow body: block or expression, bracket-aware. */
  const skipArrowBody = () => {
    ws()
    let d = 0
    if (text[i] === '{') {
      let d = 0
      while (i < len) {
        const c = text[i]
        if (c === "'" || c === '"' || c === '`') { skipString(c); continue }
        if (c === '{') d++
        else if (c === '}') { d--; if (d === 0) { i++; return } }
        i++
      }
      throw new Error('unterminated function body')
    }
    while (i < len) {
      const c = text[i]
      if (c === "'" || c === '"' || c === '`') { skipString(c); continue }
      // A comma at bracket depth 0 ends the expression leaf; brackets must be
      // tracked so array literals / ternaries / nested calls don't end early.
      if ('([{'.includes(c)) { d++; i++; continue }
      if (')]}'.includes(c)) { if (d === 0) return; d--; i++; continue }
      if (c === ',' && d === 0) return
      i++
    }
  }

  /**
   * Consume a function-ish leaf. Handles `(a, b) => body`, `x => body`,
   * `async x => body` and opaque call expressions; a bare identifier with no
   * parens and no arrow (e.g. `fieldLabels: FIELD_LABELS`) is an
   * imported-constant reference — opaque to a scanner, so locales may inline
   * it or skip it wholesale (see compare()).
   */
  const parseFnLeaf = () => {
    let sawParen = false
    let sawString = false
    for (;;) {
      while (i < len && /[A-Za-z0-9_$]/.test(text[i])) i++
      ws()
      if (text.slice(i, i + 2) === '=>') {
        i += 2
        skipArrowBody()
        return { fn: true }
      }
      if (text[i] !== '(') {
        return sawParen || sawString ? { fn: true } : { ref: true }
      }
      // Consume a balanced paren group, string-aware.
      sawParen = true
      let d = 0
      while (i < len) {
        const c = text[i]
        if (c === "'" || c === '"' || c === '`') { skipString(c); sawString = true; continue }
        if (c === '(') d++
        else if (c === ')') { d--; i++; if (d === 0) break; continue }
        i++
      }
      ws()
      if (text.slice(i, i + 2) === '=>') {
        i += 2
        skipArrowBody()
        return { fn: true }
      }
      // Parenthesized value with no arrow (call expression) — opaque fn leaf.
      return { fn: true }
    }
  }

  function parseValue() {
    ws()
    const c = text[i]
    if (c === '{') return parseObj()
    if (c === '[') return parseArr()
    if (c === "'" || c === '"' || c === '`') { skipString(c); return { str: true } }
    if (c === '(') return parseFnLeaf()
    if (/[0-9]/.test(c)) {
      while (i < len && /[0-9a-fx._+-]/i.test(text[i])) i++
      return { ref: true } // number literal — opaque scalar
    }
    if (/[A-Za-z_$]/.test(c)) return parseFnLeaf()
    throw new Error(`unexpected token at offset ${i}: ${JSON.stringify(text.slice(i, i + 24))}`)
  }

  /** Skip `as <type>` assertions after a value leaf (e.g. `…} as Record<string,
   *  string>` — TS casts carry no runtime shape and are irrelevant to parity).
   *  Handles identifiers, generic angle brackets (including commas inside
   *  them), and array suffixes like `as const` / `as readonly string[]`. */
  function skipAsClauses() {
    for (;;) {
      ws()
      if (!(text.startsWith('as', i) && !/[A-Za-z0-9_$]/.test(text[i + 2] ?? ''))) return
      i += 2
      ws()
      while (i < len && /[A-Za-z0-9_$]/.test(text[i])) i++
      ws()
      if (text[i] === '<') {
        let angle = 0
        while (i < len) {
          if (text[i] === '<') angle++
          else if (text[i] === '>') {
            angle--
            if (angle === 0) { i++; break }
          }
          i++
        }
        ws()
      }
      while (text[i] === '[') { i++; ws(); if (text[i] === ']') i++; ws() }
    }
  }

  function parseObj() {
    const out = {}
    i++ // {
    for (;;) {
      ws()
      if (text[i] === '}') { i++; return out }
      // Key: quoted or bare (may include '-' for keys like 'every-15-minutes').
      let key
      if (text[i] === "'" || text[i] === '"') {
        const s = i
        skipString(text[i])
        key = text.slice(s + 1, i - 1)
      } else {
        const s = i
        while (i < len && /[A-Za-z0-9_$-]/.test(text[i])) i++
        key = text.slice(s, i)
      }
      ws()
      if (text[i] === '?') i++ // optional property marker
      ws()
      if (text[i] !== ':') throw new Error(`expected ':' after key ${key} at offset ${i}`)
      i++ // :
      out[key] = parseValue()
      skipAsClauses()
      ws()
      if (text[i] === ',') { i++; continue }
      if (text[i] === '}') { i++; return out }
      throw new Error(`expected ',' or '}' at offset ${i}: ${JSON.stringify(text.slice(i, i + 24))}`)
    }
  }

  function parseArr() {
    const out = []
    i++ // [
    for (;;) {
      ws()
      if (text[i] === ']') { i++; return out }
      out.push(parseValue())
      skipAsClauses()
      ws()
      if (text[i] === ',') { i++; continue }
      if (text[i] === ']') { i++; return out }
      throw new Error(`expected ',' or ']' at offset ${i}: ${JSON.stringify(text.slice(i, i + 24))}`)
    }
  }

  const v = parseValue()
  // Top level is an object literal → parseObj returned the real tree.
  // (Anything else — e.g. an array — has no key structure to compare.)
  return v && typeof v === 'object' && !v.fn && !v.str ? v : {}
}

/** Parse a .ts locale file into its nested tree. */
function parseTsLocale(file) {
  const src = fs.readFileSync(file, 'utf8')
  const objText = extractTopObjectText(src)
  return parseObjectLiteral(objText)
}

// ─── Comparison ─────────────────────────────────────────────────────────────

function kindOf(leaf) {
  if (leaf && typeof leaf === 'object') {
    if (leaf.ref) return 'ref'
    if (leaf.fn) return 'fn'
    return 'str'
  }
  return typeof leaf === 'function' ? 'fn' : 'str'
}

function compare(enFlat, locFlat) {
  const problems = []
  // Keys under an en-side identifier reference (e.g. `fieldLabels:
  // FIELD_LABELS`, resolved at import time from another module) are opaque:
  // a locale may inline translated content (ar.ts/fa.ts do) — skip them.
  const refPrefixes = Object.keys(enFlat)
    .filter((k) => kindOf(enFlat[k]) === 'ref')
    .map((k) => `${k}.`)
  const underRef = (k) => refPrefixes.some((p) => k.startsWith(p))

  for (const k of Object.keys(enFlat)) {
    if (kindOf(enFlat[k]) === 'ref') continue
    if (!(k in locFlat) && !underRef(k)) problems.push(`missing  ${k}`)
  }
  for (const k of Object.keys(locFlat)) {
    if (!(k in enFlat) && !underRef(k)) problems.push(`extra    ${k}`)
  }
  // Shape parity: an interpolator leaf must stay an interpolator (and a plain
  // string must stay a string), or a runtime call like t.common.tryHint(term)
  // becomes `undefined(term)` / renders "[object Object]".
  for (const k of Object.keys(enFlat)) {
    const ek = kindOf(enFlat[k])
    if (ek === 'ref' || underRef(k)) continue
    const lk = kindOf(locFlat[k])
    if (k in locFlat && ek !== lk) problems.push(`shape    ${k} (en=${ek}, locale=${lk})`)
  }
  return problems
}

// ─── CLI ────────────────────────────────────────────────────────────────────

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// The locale surfaces Hermes ships, each with its English source of truth.
// YAML backend catalogs are complete translations: every locale must carry
// every key (tests/agent/test_i18n.py pins the same invariant in pytest).
// The TS surfaces build with defineLocale(), which deep-merges over English,
// so a locale may omit untranslated keys — but it may NOT carry stale/unknown
// keys, and it must keep leaf shapes: a string leaf stays a string, an
// interpolator stays a function, or runtime renders "[object Object]".
const SURFACES = [
  { name: 'agent', dir: 'locales', en: 'en.yaml', mode: 'strict', exclude: [] },
  {
    name: 'web',
    dir: 'web/src/i18n',
    en: 'en.ts',
    mode: 'partial',
    exclude: ['define-locale.ts', 'index.ts', 'types.ts'],
  },
  {
    name: 'desktop',
    dir: 'apps/desktop/src/i18n',
    en: 'en.ts',
    mode: 'partial',
    exclude: [
      'catalog.ts',
      'define-locale.ts',
      'index.ts',
      'languages.ts',
      'plugin-i18n.ts',
      'runtime.ts',
      'types.ts',
    ],
  },
]

/** Parse both sides and compare. Throws on parse errors. */
function checkPair(enFile, locFile, { partial = false } = {}) {
  const isYaml = (f) => f.endsWith('.yaml') || f.endsWith('.yml')
  if (isYaml(enFile) !== isYaml(locFile)) {
    throw new Error('both files must be the same format (YAML or TS)')
  }
  const enTree = isYaml(enFile) ? parseYamlLocale(enFile) : parseTsLocale(enFile)
  const locTree = isYaml(locFile) ? parseYamlLocale(locFile) : parseTsLocale(locFile)
  const enFlat = flatten(enTree)
  const locFlat = flatten(locTree)
  const problems = compare(enFlat, locFlat)
  return {
    total: Object.keys(enFlat).length,
    translated: Object.keys(locFlat).length,
    missing: problems.filter((p) => p.startsWith('missing')).length,
    extra: problems.filter((p) => p.startsWith('extra')).length,
    shape: problems.filter((p) => p.startsWith('shape')).length,
    // In partial mode missing keys are tolerated (deep-merge locales).
    problems: partial ? problems.filter((p) => !p.startsWith('missing')) : problems,
  }
}

function runAll({ verbose }) {
  let failed = 0
  for (const surface of SURFACES) {
    const dir = path.join(REPO_ROOT, surface.dir)
    const enFile = path.join(dir, surface.en)
    if (!fs.existsSync(enFile)) {
      console.log(`FAIL ${surface.name}: English source not found: ${enFile}`)
      failed++
      continue
    }
    const localeFiles = fs
      .readdirSync(dir)
      .filter(
        (f) =>
          /^[a-z]{2}(-\w+)?\.(yaml|yml|ts)$/.test(f) &&
          f !== surface.en &&
          !surface.exclude.includes(f),
      )
      .sort()
    if (!localeFiles.length) {
      console.log(`FAIL ${surface.name}: no locale files found in ${dir}`)
      failed++
      continue
    }
    for (const f of localeFiles) {
      try {
        const r = checkPair(enFile, path.join(dir, f), { partial: surface.mode === 'partial' })
        const ok = r.problems.length === 0
        if (!ok) failed++
        console.log(
          `${ok ? 'ok  ' : 'FAIL'} ${surface.name}/${f}: ${r.translated}/${r.total} keys, ` +
            `${r.missing} missing, ${r.extra} extra, ${r.shape} shape` +
            (surface.mode === 'partial' && r.missing ? ' (deep-merge: missing ok)' : ''),
        )
        if (verbose && r.problems.length) console.log(r.problems.map((p) => `       ${p}`).join('\n'))
      } catch (err) {
        failed++
        console.log(`FAIL ${surface.name}/${f}: parse failed: ${err.message}`)
      }
    }
  }
  console.log(failed ? `\n${failed} locale file(s) out of parity` : '\nAll locale files in parity')
  process.exit(failed ? 1 : 0)
}

function fail(msg) {
  console.error(`error: ${msg}`)
  process.exit(2)
}

function main() {
  const argv = process.argv.slice(2)
  const verbose = argv.includes('--verbose') || argv.includes('-v')
  const partial = argv.includes('--partial')
  const fileArgs = argv.filter((a) => !a.startsWith('-'))

  if (argv.includes('--all')) {
    if (fileArgs.length) fail('--all takes no file arguments')
    runAll({ verbose })
    return
  }

  if (fileArgs.length !== 2) {
    console.error('usage: node scripts/check_i18n_parity.mjs <en-file> <translated-file> [--partial] [--verbose]')
    console.error('       node scripts/check_i18n_parity.mjs --all [--verbose]')
    console.error('files ending .yaml/.yml parse as YAML; everything else parses as a TS locale object')
    process.exit(2)
  }
  const [enFile, locFile] = fileArgs.map((f) => path.resolve(f))
  if (!fs.existsSync(enFile)) fail(`en file not found: ${enFile}`)
  if (!fs.existsSync(locFile)) fail(`locale file not found: ${locFile}`)

  let r
  try {
    r = checkPair(enFile, locFile, { partial })
  } catch (err) {
    fail(`parse failed: ${err.message}`)
  }

  console.log(
    `${locFile}: ${r.translated}/${r.total} keys` +
    `, ${r.missing} missing, ${r.extra} extra, ${r.shape} shape`,
  )
  if (verbose && r.problems.length) console.log(r.problems.join('\n'))
  process.exit(r.problems.length ? 1 : 0)
}

main()
