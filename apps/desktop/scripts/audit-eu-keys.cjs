#!/usr/bin/env node
// Audit: compare en.ts (source of truth) vs eu.ts (overrides) key paths.
// For every top-level section that eu.ts overrides, list leaf keys present in en.ts but missing in eu.ts.
const ts = require('typescript')
const fs = require('fs')
const path = require('path')

const dir = path.join(__dirname, '..', 'src', 'i18n')
const enSrc = fs.readFileSync(path.join(dir, 'en.ts'), 'utf8')
const euSrc = fs.readFileSync(path.join(dir, 'eu.ts'), 'utf8')

function collectKeys(source, varName) {
  const sf = ts.createSourceFile('x.ts', source, ts.ScriptTarget.Latest, true)
  const keys = {}
  let found = false

  function visit(node, prefix) {
    if (ts.isVariableStatement(node)) {
      for (const decl of node.declarationList.declarations) {
        if (ts.isIdentifier(decl.name) && decl.name.text === varName && decl.initializer) {
          found = true
          walkObject(decl.initializer, prefix)
        }
      }
    }
    ts.forEachChild(node, child => visit(child, prefix))
  }

  function walkObject(node, prefix) {
    if (ts.isCallExpression(node) && node.expression.getText(sf).includes('defineLocale')) {
      const arg = node.arguments[0]
      if (arg && ts.isObjectLiteralExpression(arg)) walkObject(arg, prefix)
      return
    }
    if (!ts.isObjectLiteralExpression(node)) return
    for (const prop of node.properties) {
      if (!ts.isPropertyAssignment(prop) && !ts.isShorthandPropertyAssignment(prop)) continue
      let name
      if (ts.isIdentifier(prop.name) || ts.isStringLiteral(prop.name)) {
        name = prop.name.text
      } else {
        continue
      }
      const full = prefix ? `${prefix}.${name}` : name
      keys[full] = true
      if (ts.isPropertyAssignment(prop)) {
        const init = prop.initializer
        if (ts.isObjectLiteralExpression(init)) walkObject(init, full)
        else if (ts.isCallExpression(init) && init.expression.getText(sf).includes('defineLocale')) {
          const arg = init.arguments[0]
          if (arg && ts.isObjectLiteralExpression(arg)) walkObject(arg, full)
        }
      }
    }
  }

  visit(sf, '')
  return keys
}

const enKeys = collectKeys(enSrc, 'en')
const euKeys = collectKeys(euSrc, 'eu')

// For each top-level section that eu overrides, check en-vs-eu leaf coverage.
const euTopLevels = new Set(Object.keys(euKeys).map(k => k.split('.')[0]))
const missing = []
for (const section of euTopLevels) {
  const prefix = section + '.'
  const enSection = Object.keys(enKeys).filter(k => k.startsWith(prefix))
  for (const k of enSection) {
    if (!euKeys[k]) missing.push(k)
  }
}
// Keys eu declares that en doesn't (shouldn't happen — means typo in eu):
const extra = Object.keys(euKeys).filter(k => !enKeys[k])

console.log('=== en.ts top-level sections:', Object.keys(enKeys).filter(k => !k.includes('.')).length)
console.log('=== eu.ts override sections:', [...euTopLevels].join(', '))
console.log('=== MISSING in eu.ts (present in en.ts, section overridden):')
console.log(missing.length ? missing.join('\n') : '(none)')
console.log('=== EXTRA in eu.ts (not in en.ts — possible typo):')
console.log(extra.length ? extra.join('\n') : '(none)')
