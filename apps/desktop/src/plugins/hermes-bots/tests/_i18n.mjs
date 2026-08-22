// Shared test helper: the plugin's own English bundle + translator, for the
// harnesses that eval a SLICE of plugin.js in a fresh VM context. Those slices
// reference the module-level `t` without carrying its declaration, so every
// sandboxed context needs one. Reusing the REAL bundle (rather than a
// key-echoing stub) keeps the English text assertions in these tests honest
// and exercises the plugin's English fallback path at the same time.
import { readFileSync } from 'node:fs'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function sliceBundle(name) {
  const start = source.indexOf(`const ${name} = {`)

  if (start < 0) {
    throw new Error(`${name} not found in plugin.js`)
  }

  const open = source.indexOf('{', start)
  let depth = 0

  for (let i = open; i < source.length; i++) {
    if (source[i] === '{') depth++
    else if (source[i] === '}') {
      depth--
      if (depth === 0) return source.slice(open, i + 1)
    }
  }

  throw new Error(`${name} literal is unbalanced`)
}

const EN = vm.runInNewContext(`(${sliceBundle('EN_MESSAGES')})`)

/** Mirrors the plugin's own resolver: dot-path lookup, functions called with
 *  the args, unknown keys echoed back. */
export function t(key, ...args) {
  const value = String(key)
    .split('.')
    .reduce((node, part) => (node == null ? undefined : node[part]), EN)

  if (typeof value === 'function') {
    return value(...args)
  }

  return typeof value === 'string' ? value : key
}
