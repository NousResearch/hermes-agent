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
    if (id.includes('settings/constants')) return { FIELD_LABELS: { __raw: 'FIELD_LABELS' }, FIELD_DESCRIPTIONS: { __raw: 'FIELD_DESCRIPTIONS' } }
    if (id.includes('field-copy')) return { defineFieldCopy: (o) => o }
    if (id.includes('define-locale')) return { defineLocale: (o) => o }
    if (id.includes('lib/')) return {}
    return {}
  }
  new Function('require', 'module', 'exports', out)(req, m, m.exports)
  if (isOv) return m.exports.csOverrides || m.exports.cs
  return m.exports.en
}

function flat(o, p, a) {
  p = p || []; a = a || {}
  if (o && typeof o === 'object' && !Array.isArray(o) && '__raw' in o) { a[p.join('.')] = { kind: 'raw', v: o.__raw }; return a }
  if (typeof o === 'string') { a[p.join('.')] = { kind: 'string', v: o }; return a }
  if (typeof o === 'function') { a[p.join('.')] = { kind: 'function', v: o }; return a }
  if (Array.isArray(o)) { a[p.join('.')] = { kind: 'array', v: o }; return a }
  if (o && typeof o === 'object') { for (const k of Object.keys(o)) flat(o[k], p.concat(k), a); return a }
  a[p.join('.')] = { kind: '?', v: o }; return a
}

const en = flat(load(path.join(ROOT, 'apps/desktop/src/i18n/en.ts'), false))
const cs = flat(load(path.join(ROOT, 'apps/desktop/src/i18n/cs.ts'), true))

const art = (k) => k.startsWith('settings.fieldLabels') || k.startsWith('settings.fieldDescriptions')
const miss = Object.keys(en).filter((k) => !(k in cs) && !art(k) && !k.startsWith('intro.'))
const obs = Object.keys(cs).filter((k) => !(k in en) && !art(k) && !k.startsWith('intro.'))

const mode = process.argv[2] || 'summary'

if (mode === 'details') {
  for (const k of miss) {
    const e = en[k]
    if (e.kind === 'function') console.log(`FUNC ${k} :: ${String(e.v).replace(/\s+/g, ' ')}`)
    else if (e.kind === 'string') console.log(`STR  ${k} :: ${JSON.stringify(e.v)}`)
    else if (e.kind === 'array') console.log(`ARR  ${k} :: ${JSON.stringify(e.v)}`)
    else console.log(`${e.kind.toUpperCase()}  ${k} :: ${JSON.stringify(e.v)}`)
  }
} else {
  console.log('MISSING', miss.length, 'OBSOLETE', obs.length)
  console.log(miss.join('\n'))
  if (obs.length) console.log('OBSOLETE:\n' + obs.join('\n'))
}

// ---- field copy coverage against the real constants.ts (AST walk) ----
function keysOfConst(name) {
  const file = path.join(ROOT, 'apps/desktop/src/app/settings/constants.ts')
  const src = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.ES2020, true)
  const found = []
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText() === name && node.initializer) {
      let init = node.initializer
      // unwrap calls like defineFieldCopy({...})
      const unwrap = (n) => (ts.isCallExpression(n) ? unwrap(n.arguments[0]) : n)
      init = unwrap(init)
      if (init && ts.isObjectLiteralExpression(init)) {
        for (const p of init.properties) {
          if (ts.isPropertyAssignment(p)) found.push(p.name.getText().replace(/['"]/g, ''))
        }
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(src)
  return found
}

const labels = keysOfConst('FIELD_LABELS')
const descs = keysOfConst('FIELD_DESCRIPTIONS')
const csLabelKeys = Object.keys(cs).filter((k) => k.startsWith('settings.fieldLabels.')).map((k) => k.slice('settings.fieldLabels.'.length))
const csDescKeys = Object.keys(cs).filter((k) => k.startsWith('settings.fieldDescriptions.')).map((k) => k.slice('settings.fieldDescriptions.'.length))
console.log('--- fieldLabels: constants=', labels.length, ' cs=', csLabelKeys.length, ' missing=', labels.filter((k) => !csLabelKeys.includes(k)).length)
console.log(labels.filter((k) => !csLabelKeys.includes(k)).join('\n'))
console.log('--- fieldDescriptions: constants=', descs.length, ' cs=', csDescKeys.length, ' missing=', descs.filter((k) => !csDescKeys.includes(k)).length)
console.log(descs.filter((k) => !csDescKeys.includes(k)).join('\n'))
console.log('--- cs field keys not in constants (obsolete):', csLabelKeys.filter((k) => !labels.includes(k)).concat(csDescKeys.filter((k) => !descs.includes(k))).join(', '))
