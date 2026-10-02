const fs = require('fs')
const path = require('path')
const ts = require('/root/hermes-agent/node_modules/typescript')

const ROOT = '/tmp/sync'

function load(file, isOv) {
  const src = fs.readFileSync(file, 'utf8')
  const out = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 }
  }).outputText
  const m = { exports: {} }
  const req = (id) => {
    if (id.includes('field-copy')) return { defineFieldCopy: (o) => o }
    if (id.includes('settings/constants')) return { FIELD_LABELS: { __raw: 'FIELD_LABELS' }, FIELD_DESCRIPTIONS: { __raw: 'FIELD_DESCRIPTIONS' } }
    if (id.includes('define-locale')) return { defineLocale: (o) => o }
    return {}
  }
  new Function('require', 'module', 'exports', out)(req, m, m.exports)
  if (isOv) return m.exports.csOverrides || m.exports.cs
  return m.exports.en
}

function flat(o, p, a) {
  p = p || []; a = a || {}
  if (o && typeof o === 'object' && !Array.isArray(o) && '__raw' in o) { a[p.join('.')] = { kind: 'raw' }; return a }
  if (typeof o === 'string') { a[p.join('.')] = { kind: 'string', v: o }; return a }
  if (typeof o === 'function') { a[p.join('.')] = { kind: 'function', v: o }; return a }
  if (Array.isArray(o)) { a[p.join('.')] = { kind: 'array', v: o }; return a }
  if (o && typeof o === 'object') { for (const k of Object.keys(o)) flat(o[k], p.concat(k), a); return a }
  a[p.join('.')] = { kind: '?' }; return a
}

function unwrap(node) {
  if (ts.isCallExpression(node)) return unwrap(node.arguments[0])
  if (ts.isParenthesizedExpression(node) || ts.isAsExpression(node) || ts.isSatisfiesExpression(node)) return unwrap(node.expression)
  return node
}

function keysOfConst(name) {
  const file = path.join(ROOT, 'apps/desktop/src/app/settings/constants.ts')
  const src = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.ES2020, true)
  const out = []
  const walk = (node, prefix) => {
    const init = unwrap(node)
    if (!init || !ts.isObjectLiteralExpression(init)) return
    for (const p of init.properties) {
      if (!ts.isPropertyAssignment(p)) continue
      const key = p.name.getText().replace(/^['"]|['"]$/g, '')
      const full = prefix ? prefix + '.' + key : key
      const val = unwrap(p.initializer)
      if (val && ts.isObjectLiteralExpression(val)) walk(val, full)
      else out.push(full)
    }
  }
  const visit = (node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText() === name && node.initializer) walk(node.initializer, '')
    ts.forEachChild(node, visit)
  }
  visit(src)
  return out
}

const cs = flat(load(path.join(ROOT, 'apps/desktop/src/i18n/cs.ts'), true))
const constLabels = keysOfConst('FIELD_LABELS')
const constDescs = keysOfConst('FIELD_DESCRIPTIONS')
const csLabels = Object.keys(cs).filter((k) => k.startsWith('settings.fieldLabels.')).map((k) => k.slice(21))
const csDescs = Object.keys(cs).filter((k) => k.startsWith('settings.fieldDescriptions.')).map((k) => k.slice(27))

const rep = (label, constKeys, csKeys) => {
  const missing = constKeys.filter((k) => !csKeys.includes(k))
  const extra = csKeys.filter((k) => !constKeys.includes(k))
  console.log(`--- ${label}: constants=${constKeys.length} cs=${csKeys.length} missing=${missing.length} extra=${extra.length}`)
  if (missing.length) console.log('MISSING: ' + missing.join(', '))
  if (extra.length) console.log('EXTRA: ' + extra.join(', '))
}

rep('FIELD_LABELS', constLabels, csLabels)
rep('FIELD_DESCRIPTIONS', constDescs, csDescs)
