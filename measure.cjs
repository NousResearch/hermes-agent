const fs = require('fs')
const path = require('path')
const ts = require('/root/hermes-agent/node_modules/typescript')
const ROOT = '/tmp/sync'

function load(file, isOv) {
  const src = fs.readFileSync(file, 'utf8')
  const out = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
  const m = { exports: {} }
  const req = (id) => {
    if (id.includes('field-copy')) return { defineFieldCopy: (o) => o }
    if (id.includes('settings/constants')) return { FIELD_LABELS: { __raw: 'FIELD_LABELS' }, FIELD_DESCRIPTIONS: { __raw: 'FIELD_DESCRIPTIONS' } }
    if (id.includes('define-locale')) return { defineLocale: (o) => o }
    return {}
  }
  new Function('require', 'module', 'exports', out)(req, m, m.exports)
  return isOv ? (m.exports.csOverrides || m.exports.cs) : m.exports.en
}

function flat(o, p, a) {
  p = p || []; a = a || {}
  if (o && typeof o === 'object' && !Array.isArray(o) && '__raw' in o) { a[p.join('.')] = { kind: 'raw' }; return a }
  if (typeof o === 'string' || typeof o === 'function' || Array.isArray(o)) { a[p.join('.')] = { kind: typeof o === 'string' ? 'string' : typeof o === 'function' ? 'function' : 'array', v: o }; return a }
  if (o && typeof o === 'object') { for (const k of Object.keys(o)) flat(o[k], p.concat(k), a); return a }
  a[p.join('.')] = { kind: '?' }; return a
}

const en = flat(load(path.join(ROOT, 'apps/desktop/src/i18n/en.ts'), false))
const cs = flat(load(path.join(ROOT, 'apps/desktop/src/i18n/cs.ts'), true))
const art = (k) => k.startsWith('settings.fieldLabels') || k.startsWith('settings.fieldDescriptions')
const enk = Object.keys(en).filter((k) => !art(k) && !k.startsWith('intro.'))
const csk = Object.keys(cs).filter((k) => !k.startsWith('intro.'))
const fieldN = csk.filter(art).length
console.log('desktop EN(no field tree)=' + enk.length + '  CS(incl. field tree)=' + csk.length + '  fieldEntries=' + fieldN)
console.log('missing=' + enk.filter((k) => !(k in cs)).length + ' obsolete=' + Object.keys(cs).filter((k) => !(k in en) && !art(k) && !k.startsWith('intro.')).length)
const identical = enk.filter((k) => k in cs && en[k].kind === 'string' && cs[k].kind === 'string' && en[k].v === cs[k].v)
console.log('byte-identical string values (EN, no field tree): ' + identical.length)
console.log(identical.slice(0, 40).join('\n'))
