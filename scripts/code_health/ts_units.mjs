// TypeScript structure for the code-health ratchet: per-function CC, length, nesting.
// Parses only (no type check), so it is fast and needs nothing but the `typescript` package.
//
//   node ts_units.mjs <typescript-module-path> <root>  < paths.json  > units.json
//
// CC is classic McCabe, like ruff's: 1 + if, ?:, case, loops, catch, &&, ||, ??.
// It is NOT ESLint's `complexity` rule, which also counts these (so ESLint's numbers are
// higher on the same code): optional chaining (`a?.b`, `f?.()`), the logical assignments
// `&&=`, `||=`, `??=`, and default parameter values (`f(x = 1)`). They are left out on purpose,
// to stay comparable with Python's count; the ratchet compares this count only with itself.
//
// Identity: a unit's name is its path of classes, functions and named object literals
// (`Cls.method`, `outer.inner`, `obj.method` for `const obj = { method() {} }`); a repeated
// name gets `#2`, `#3` in source order, and the comparison pairs repeated names by body, not
// by that number. Signatures without a body (overloads, `declare`, abstract) are not units.
// The body hash covers the token stream (comments and whitespace excluded), so a renamed or
// re-commented function is still the same code. A file with syntax errors is an error, never
// a partial measurement.
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

// The file's code tokens in order (comments, whitespace and JSDoc excluded), collected once:
// a function's hash input is its slice of them, so a comment edit is not a code edit.
function fileTokens(sf) {
  const starts = []
  const texts = []
  const visit = n => {
    if (n.kind >= K.FirstJSDocNode && n.kind <= K.LastJSDocNode) return
    const children = n.getChildren(sf)
    if (children.length === 0) {
      const text = ts.isJsxText(n) ? n.getText(sf).replace(/\s+/g, ' ').trim() : n.getText(sf)
      if (text) {
        starts.push(n.getStart(sf))
        texts.push(text)
      }
      return
    }
    for (const child of children) visit(child)
  }
  visit(sf)
  return { starts, texts }
}

function lowerBound(sorted, value) {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] < value) lo = mid + 1
    else hi = mid
  }
  return lo
}

// The tokens of `node`, without those starting at a position in `cuts`.
function tokenText(tokens, node, sf, cuts) {
  const out = []
  const end = lowerBound(tokens.starts, node.getEnd())
  for (let i = lowerBound(tokens.starts, node.getStart(sf)); i < end; i++) {
    if (!cuts.has(tokens.starts[i])) out.push(tokens.texts[i])
  }
  return out.join(' ')
}

// Whether `n` declares `name` (a variable, parameter, function or class of that name).
function declaresName(n, name) {
  const declares = ts.isVariableDeclaration(n) || ts.isParameter(n) ||
    ts.isFunctionDeclaration(n) || ts.isClassDeclaration(n)
  return Boolean(declares && n.name && ts.isIdentifier(n.name) && n.name.text === name)
}

// Whether function `fn`'s own scope binds `name`: a parameter, a named function expression's
// own name, or a declaration in its body outside any nested function or class (those bind
// only inside themselves).
function bindsOwn(fn, name) {
  if (fn.parameters?.some(p => declaresName(p, name))) return true
  if (ts.isFunctionExpression(fn) && fn.name?.text === name) return true
  let found = false
  const walk = n => {
    if (found) return
    if (declaresName(n, name)) found = true
    else if (!ts.isFunctionLike(n) && !ts.isClassLike(n)) ts.forEachChild(n, walk)
  }
  if (fn.body) walk(fn.body)
  return found
}

// The body's tokens with its own references to `name` removed (`name(...)`, `this.name(...)`),
// unless the function's own scope rebinds that name: renaming a recursive function together
// with its self-call is still the same code. A nested function that binds the name
// (`.map((legacy) => legacy)`) keeps its own reads. Strings and other objects' `.name`
// members are untouched.
function bodyWithoutSelf(tokens, node, sf, name) {
  if (!node.body) return ''
  const cuts = new Set()
  if (!name || bindsOwn(node, name)) return tokenText(tokens, node.body, sf, cuts)
  const visit = n => {
    if (n !== node.body && ts.isFunctionLike(n) && bindsOwn(n, name)) return
    if (ts.isIdentifier(n) && n.text === name) {
      const p = n.parent
      const member = p && ts.isPropertyAccessExpression(p) && p.name === n
      const key = p && (ts.isPropertyAssignment(p) || ts.isMethodDeclaration(p) ||
        ts.isPropertyDeclaration(p)) && p.name === n
      if ((!member && !key) || (member && p.expression.kind === K.ThisKeyword)) {
        cuts.add(n.getStart(sf))
      }
    }
    ts.forEachChild(n, visit)
  }
  visit(node.body)
  return tokenText(tokens, node.body, sf, cuts)
}

// The name an object literal is bound to (`const a = {...}`, `a: {...}`), so its methods are
// `a.f` and `b.f`, not `f` and `f#2` paired by position. Unbound objects add nothing.
function objectName(node) {
  let current = node.parent
  while (current && (ts.isParenthesizedExpression(current) || ts.isAsExpression(current) ||
         ts.isSatisfiesExpression?.(current) || ts.isTypeAssertionExpression(current))) {
    current = current.parent
  }
  if (current && (ts.isVariableDeclaration(current) || ts.isPropertyAssignment(current) ||
      ts.isPropertyDeclaration(current))) return propertyName(current.name)
  return null
}

function parseError(sf) {
  const diag = sf.parseDiagnostics?.[0]
  if (!diag) return null
  const line = sf.getLineAndCharacterOfPosition(diag.start ?? 0).line + 1
  return `does not parse: ${ts.flattenDiagnosticMessageText(diag.messageText, ' ')} (line ${line})`
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
  // The parser recovers from syntax errors instead of throwing: a broken file must fail
  // closed (MEASURE), not read as whatever functions survived recovery.
  const error = parseError(sf)
  if (error) return { error }
  const tokens = fileTokens(sf)
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
    } else if (ts.isObjectLiteralExpression(node)) {
      const name = objectName(node)
      if (name) next = [...stack, name]
    } else if (FUNCTION_KINDS.has(node.kind) && node.body) {
      const qual = unique([...stack, ownName(node) ?? '<anon>'].join('.'))
      const start = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1
      const end = sf.getLineAndCharacterOfPosition(node.getEnd()).line + 1
      // Name-independent: parameters + body only, so a rename (or a move) keeps the cap.
      const params = node.parameters.map(p => tokenText(tokens, p, sf, new Set())).join(' , ')
      const selfName = propertyName(node.name) ?? bindingName(node)
      const body = params + ' => ' + bodyWithoutSelf(tokens, node, sf, selfName)
      const unit = {
        q: qual, line: start, end, cc: complexity(node), nesting: nesting(node),
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
