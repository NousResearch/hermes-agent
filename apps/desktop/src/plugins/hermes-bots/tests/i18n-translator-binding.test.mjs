// Two ways the translator silently stops translating, both of which shipped in
// this plugin before this test existed:
//
//   1. a local binding named `t` shadows it, so t('key') hits that value
//      (TypeError at runtime, or a ReferenceError above the declaration);
//   2. a t() call is evaluated at module load, freezing the English fallback
//      because register() has not swapped in ctx.i18n yet.
//
// Both are invisible to the type checker and to every render test that runs in
// English, so they are asserted against the source directly.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const file = new URL('../plugin.js', import.meta.url)
const source = readFileSync(file, 'utf8')
const sf = ts.createSourceFile('plugin.js', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS)
const lineOf = node => sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1

const isTranslatorCall = node =>
  ts.isCallExpression(node) &&
  ts.isIdentifier(node.expression) &&
  node.expression.text === 't' &&
  node.arguments.length > 0 &&
  ts.isStringLiteral(node.arguments[0])

test('no local binding named `t` covers a translator call', () => {
  const shadows = []

  const collect = node => {
    const named =
      (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text === 't') ||
      (ts.isParameter(node) && ts.isIdentifier(node.name) && node.name.text === 't')

    if (named) {
      let scope = node.parent

      while (scope && !ts.isBlock(scope) && !ts.isSourceFile(scope) && !ts.isArrowFunction(scope)) {
        scope = scope.parent
      }

      if (scope && !ts.isSourceFile(scope)) {
        shadows.push({ line: lineOf(node), start: scope.getStart(sf), end: scope.getEnd() })
      }
    }

    ts.forEachChild(node, collect)
  }

  collect(sf)

  const calls = []
  const collectCalls = node => {
    if (isTranslatorCall(node)) calls.push({ line: lineOf(node), pos: node.getStart(sf), text: node.getText(sf) })
    ts.forEachChild(node, collectCalls)
  }
  collectCalls(sf)

  assert.ok(calls.length > 100, `expected the plugin to call t() widely, saw ${calls.length}`)

  const collisions = shadows.flatMap(s =>
    calls
      .filter(c => c.pos > s.start && c.pos < s.end)
      .map(c => `line ${c.line}: ${c.text} sits inside the scope of the \`t\` bound on line ${s.line}`)
  )

  assert.deepEqual(collisions, [], 'rename the local binding; `t` belongs to the translator')
})

test('no t() call is evaluated at module load', () => {
  const early = []

  const walk = (node, insideFn) => {
    const entersFn =
      ts.isFunctionDeclaration(node) ||
      ts.isFunctionExpression(node) ||
      ts.isArrowFunction(node) ||
      ts.isMethodDeclaration(node) ||
      ts.isGetAccessor(node) ||
      ts.isSetAccessor(node)

    if (!insideFn) {
      if (isTranslatorCall(node)) {
        early.push(`line ${lineOf(node)}: ${node.getText(sf)}`)
      }

      if (
        ts.isVariableDeclaration(node) &&
        node.initializer &&
        ts.isIdentifier(node.initializer) &&
        node.initializer.text === 't'
      ) {
        early.push(`line ${lineOf(node)}: captures \`t\` by reference`)
      }
    }

    ts.forEachChild(node, child => walk(child, insideFn || entersFn))
  }

  walk(sf, false)

  assert.deepEqual(
    early,
    [],
    'module-level t() runs before register() wires ctx.i18n, so it freezes the English fallback — build the value lazily'
  )
})
