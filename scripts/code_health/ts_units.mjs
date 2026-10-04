// TypeScript structure for the code-health ratchet: per-function CC, length, nesting.
// Parses only (no type check), so it is fast and needs nothing but the `typescript` package.
//
//   node ts_units.mjs <typescript-module-path> <root>  < paths.json  > units.json
//
// CC is classic McCabe, like ruff's: 1 + if, ?:, case, loops, catch, &&, ||, ??.
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join } from 'node:path'

const [tsPath, root] = process.argv.slice(2)
const ts = createRequire(import.meta.url)(tsPath)
const K = ts.SyntaxKind

const FUNCTION_KINDS = new Set([
  K.FunctionDeclaration, K.MethodDeclaration, K.ArrowFunction, K.FunctionExpression,
  K.Constructor, K.GetAccessor, K.SetAccessor
])
const BRANCH_KINDS = new Set([
  K.IfStatement, K.ConditionalExpression, K.ForStatement, K.ForInStatement, K.ForOfStatement,
  K.WhileStatement, K.DoStatement, K.CatchClause
])
const BLOCK_KINDS = new Set([
  K.IfStatement, K.ForStatement, K.ForInStatement, K.ForOfStatement, K.WhileStatement,
  K.DoStatement, K.SwitchStatement, K.TryStatement
])
const LOGICAL_OPS = new Set([
  K.AmpersandAmpersandToken, K.BarBarToken, K.QuestionQuestionToken
])

function complexity(fn) {
  let cc = 1
  const visit = node => {
    if (node !== fn && FUNCTION_KINDS.has(node.kind)) return
    if (BRANCH_KINDS.has(node.kind)) cc++
    else if (node.kind === K.CaseClause) cc++
    else if (node.kind === K.BinaryExpression && LOGICAL_OPS.has(node.operatorToken.kind)) cc++
    ts.forEachChild(node, visit)
  }
  visit(fn)
  return cc
}

function nesting(fn) {
  let deepest = 0
  const visit = (node, depth) => {
    if (node !== fn && FUNCTION_KINDS.has(node.kind)) return
    let next = depth
    // `else if` stays at the depth of its `if`.
    const isElseIf = node.kind === K.IfStatement && node.parent?.kind === K.IfStatement &&
      node.parent.elseStatement === node
    if (BLOCK_KINDS.has(node.kind) && !isElseIf) {
      next = depth + 1
      deepest = Math.max(deepest, next)
    }
    ts.forEachChild(node, child => visit(child, next))
  }
  visit(fn, 0)
  return deepest
}

function propertyName(name) {
  if (!name) return null
  if (ts.isIdentifier(name) || ts.isPrivateIdentifier(name) || ts.isStringLiteral(name) ||
      ts.isNumericLiteral(name)) return name.text
  return null
}

// Name an anonymous function from where it is bound: `const x = () => {}`,
// `x: () => {}`, `const x = useCallback(() => {})`.
function bindingName(node) {
  let current = node.parent
  while (current && (ts.isCallExpression(current) || ts.isParenthesizedExpression(current) ||
         ts.isAsExpression(current) || ts.isSatisfiesExpression?.(current))) {
    if (ts.isCallExpression(current) && current.parent && ts.isExpressionStatement(current.parent)) {
      return null
    }
    current = current.parent
  }
  if (!current) return null
  if (ts.isVariableDeclaration(current) || ts.isPropertyAssignment(current) ||
      ts.isPropertyDeclaration(current)) return propertyName(current.name)
  return null
}

function ownName(node) {
  if (node.kind === K.Constructor) return 'constructor'
  const named = propertyName(node.name)
  if (named) return node.kind === K.GetAccessor ? `get ${named}` : node.kind === K.SetAccessor ? `set ${named}` : named
  return bindingName(node)
}

// The body's text with its own references to `name` removed (`name(...)`, `this.name(...)`),
// unless the body rebinds that name: renaming a recursive function together with its
// self-call is still the same code. Strings and other objects' `.name` members are untouched.
function bodyWithoutSelf(node, sf, name) {
  if (!node.body) return ''
  const start = node.body.getStart(sf)
  let text = node.body.getText(sf)
  if (!name) return text
  const cuts = []
  let rebound = node.parameters.some(p => ts.isIdentifier(p.name) && p.name.text === name)
  const visit = n => {
    const declares = ts.isVariableDeclaration(n) || ts.isParameter(n) ||
      ts.isFunctionDeclaration(n) || ts.isClassDeclaration(n)
    if (declares && n.name && ts.isIdentifier(n.name) && n.name.text === name) rebound = true
    if (ts.isIdentifier(n) && n.text === name) {
      const p = n.parent
      const member = p && ts.isPropertyAccessExpression(p) && p.name === n
      const key = p && (ts.isPropertyAssignment(p) || ts.isMethodDeclaration(p) ||
        ts.isPropertyDeclaration(p)) && p.name === n
      if ((!member && !key) || (member && p.expression.kind === K.ThisKeyword)) {
        cuts.push(n.getStart(sf) - start)
      }
    }
    ts.forEachChild(n, visit)
  }
  visit(node.body)
  if (rebound) return text
  for (const at of cuts.sort((a, b) => b - a)) text = text.slice(0, at) + text.slice(at + name.length)
  return text
}

// Comment trivia (not string or template text) that carries a `health: allow` directive.
function allowComments(sf, text) {
  const found = new Map()
  const collect = ranges => {
    for (const range of ranges ?? []) {
      const body = text.slice(range.pos, range.end)
      if (body.includes('health:')) {
        found.set(range.pos, [sf.getLineAndCharacterOfPosition(range.pos).line + 1, body])
      }
    }
  }
  const visit = node => {
    collect(ts.getLeadingCommentRanges(text, node.pos))
    collect(ts.getTrailingCommentRanges(text, node.end))
    ts.forEachChild(node, visit)
  }
  visit(sf)
  collect(ts.getLeadingCommentRanges(text, sf.endOfFileToken.pos))
  return [...found.values()]
}

function measureFile(path) {
  const text = readFileSync(join(root, path), 'utf8')
  const kind = path.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  const sf = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, kind)
  const units = []
  const seen = new Map()
  const unique = q => {
    const n = (seen.get(q) ?? 0) + 1
    seen.set(q, n)
    return n === 1 ? q : `${q}#${n}`
  }
  // `lines` is a function's OWN lines: nested functions are their own units, so editing a
  // closure inside a hook never counts against the hook itself. A line is owned by the
  // innermost function that covers it, so siblings sharing a line give it up once, not twice.
  const owner = new Map() // line -> unit
  const walk = (node, stack, parent) => {
    let next = stack
    let nextParent = parent
    if (ts.isClassDeclaration(node) || ts.isClassExpression(node)) {
      next = [...stack, propertyName(node.name) ?? '<class>']
    } else if (FUNCTION_KINDS.has(node.kind)) {
      const qual = unique([...stack, ownName(node) ?? '<anon>'].join('.'))
      const start = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1
      const end = sf.getLineAndCharacterOfPosition(node.getEnd()).line + 1
      // Name-independent: parameters + body only, so a rename (or a move) keeps the cap.
      const params = node.parameters.map(p => p.getText(sf)).join(',')
      const selfName = propertyName(node.name) ?? bindingName(node)
      const body = (params + '=>' + bodyWithoutSelf(node, sf, selfName)).replace(/\s+/g, ' ')
      const unit = {
        q: qual, line: start, cc: complexity(node), nesting: nesting(node),
        hash: createHash('sha1').update(body).digest('hex').slice(0, 16)
      }
      for (let line = start; line <= end; line++) owner.set(line, unit)
      units.push(unit)
      next = [...stack, qual.split('.').pop()]
      nextParent = unit
    }
    ts.forEachChild(node, child => walk(child, next, nextParent))
  }
  walk(sf, [], null)
  for (const unit of units) unit.lines = 0
  for (const unit of owner.values()) unit.lines++
  return { units, comments: allowComments(sf, text) }
}

const paths = JSON.parse(readFileSync(0, 'utf8'))
const out = {}
for (const path of paths) {
  try {
    out[path] = measureFile(path)
  } catch (error) {
    out[path] = { error: String(error) }
  }
}
process.stdout.write(JSON.stringify(out))
