/**
 * Inverse-compile `jsx()` / `jsxs()` runtime calls back into JSX syntax.
 *
 * hermes-bots/plugin.js is hand-written JSX compiler output: it imports
 * { jsx, jsxs } from 'react/jsx-runtime' and calls them directly. That makes
 * the conversion to real JSX a deterministic AST transform rather than a
 * rewrite.
 *
 * Usage: node dejsx.mjs <in.js> <out.tsx>
 */
import { readFileSync, writeFileSync } from 'node:fs'

import generate from '@babel/generator'
import { parse } from '@babel/parser'
import traverse from '@babel/traverse'
import * as t from '@babel/types'

const gen = generate.default ?? generate
const walk = traverse.default ?? traverse

const [, , inPath, outPath] = process.argv

const source = readFileSync(inPath, 'utf8')
const ast = parse(source, {
  sourceType: 'module',
  plugins: ['jsx'],
  attachComment: true
})

const stats = { jsx: 0, jsxs: 0, skipped: [] }

/** `'div'` → intrinsic; `Foo` → component; `Foo.Bar` → member. */
function toJsxName(node) {
  if (t.isStringLiteral(node)) {
    // Only bare tag names are expressible as JSX intrinsics.
    return /^[a-z][a-z0-9]*$/i.test(node.value) ? t.jsxIdentifier(node.value) : null
  }

  if (t.isIdentifier(node)) {
    return t.jsxIdentifier(node.name)
  }

  if (t.isMemberExpression(node) && !node.computed) {
    const object = toJsxName(node.object)
    const property = t.isIdentifier(node.property) ? t.jsxIdentifier(node.property.name) : null

    return object && property ? t.jsxMemberExpression(object, property) : null
  }

  return null
}

/** JSX attribute names permit dashes and colons, so most keys pass through. */
function toAttrName(key, computed) {
  if (computed) {
    return null
  }

  const raw = t.isIdentifier(key) ? key.name : t.isStringLiteral(key) ? key.value : null

  return raw && /^[A-Za-z_$][-:A-Za-z0-9_$]*$/.test(raw) ? t.jsxIdentifier(raw) : null
}

/**
 * A string child is safe as bare JSXText only when it survives a round trip:
 * no JSX metacharacters, and no leading/trailing whitespace (which JSX trims).
 */
function textIsSafe(value) {
  return value.length > 0 && !/[{}<>]/.test(value) && value === value.trim() && !/\n/.test(value)
}

function toChild(node) {
  if (t.isJSXElement(node) || t.isJSXFragment(node)) {
    return node
  }

  if (t.isStringLiteral(node) && textIsSafe(node.value)) {
    return t.jsxText(node.value)
  }

  // `null` / `false` children are pure noise once inlined, but they carry
  // meaning inside conditionals, so only drop the standalone literals.
  if (t.isNullLiteral(node) || (t.isBooleanLiteral(node) && node.value === false)) {
    return null
  }

  return t.jsxExpressionContainer(node)
}

/**
 * Comments between children have to become `{/* … *␟/}` — a bare `//` in JSX
 * children position is literal text, not a comment. Line comments are
 * rewritten as block comments so they survive the move.
 */
function commentChild(comments) {
  const empty = t.jsxEmptyExpression()

  empty.innerComments = comments.map(comment => ({
    type: 'CommentBlock',
    value: comment.type === 'CommentLine' ? ` ${comment.value.trim()} ` : comment.value
  }))

  return t.jsxExpressionContainer(empty)
}

function childrenFrom(node) {
  const items = t.isArrayExpression(node) ? node.elements : [node]
  const children = []

  for (const item of items) {
    if (!item) {
      continue
    }

    if (item.leadingComments?.length) {
      children.push(commentChild(item.leadingComments))
      item.leadingComments = null
    }

    const child = t.isSpreadElement(item) ? t.jsxExpressionContainer(item.argument) : toChild(item)

    if (child) {
      children.push(child)
    }

    if (item.trailingComments?.length) {
      children.push(commentChild(item.trailingComments))
      item.trailingComments = null
    }
  }

  return children
}

function convert(path) {
  const { node } = path
  const [type, props, key] = node.arguments

  const name = toJsxName(type)

  if (!name) {
    stats.skipped.push(`${node.loc?.start.line}: dynamic element type`)

    return
  }

  if (!t.isObjectExpression(props)) {
    stats.skipped.push(`${node.loc?.start.line}: non-literal props`)

    return
  }

  const attributes = []
  let children = []

  if (key) {
    attributes.push(t.jsxAttribute(t.jsxIdentifier('key'), t.jsxExpressionContainer(key)))
  }

  for (const prop of props.properties) {
    if (t.isSpreadElement(prop)) {
      attributes.push(t.jsxSpreadAttribute(prop.argument))

      continue
    }

    if (!t.isObjectProperty(prop)) {
      stats.skipped.push(`${node.loc?.start.line}: object method in props`)

      return
    }

    const attrName = toAttrName(prop.key, prop.computed)

    if (!attrName) {
      stats.skipped.push(`${node.loc?.start.line}: unexpressible prop key`)

      return
    }

    if (attrName.name === 'children') {
      children = childrenFrom(prop.value)

      continue
    }

    // `foo={true}` is idiomatic as a bare `foo`; string values print unquoted.
    const value =
      t.isStringLiteral(prop.value) && !/[\n"]/.test(prop.value.value)
        ? t.stringLiteral(prop.value.value)
        : t.isBooleanLiteral(prop.value) && prop.value.value === true
          ? null
          : t.jsxExpressionContainer(prop.value)

    const attribute = t.jsxAttribute(attrName, value)

    // Carry the explanatory comments that sit above props — this file's
    // comments are most of its documentation.
    if (prop.leadingComments?.length) {
      attribute.leadingComments = prop.leadingComments
    }

    attributes.push(attribute)
  }

  const selfClosing = children.length === 0
  const element = t.jsxElement(
    t.jsxOpeningElement(name, attributes, selfClosing),
    selfClosing ? null : t.jsxClosingElement(name),
    children,
    selfClosing
  )

  t.inherits(element, node)
  path.replaceWith(element)

  stats[node.callee.name] += 1
}

walk(ast, {
  CallExpression: {
    // Post-order: inner calls are already JSXElements by the time we rebuild
    // the parent, so they slot straight in as children.
    exit(path) {
      const callee = path.node.callee

      if (t.isIdentifier(callee) && (callee.name === 'jsx' || callee.name === 'jsxs')) {
        convert(path)
      }
    }
  }
})

// The runtime import is what we just eliminated.
walk(ast, {
  ImportDeclaration(path) {
    if (path.node.source.value === 'react/jsx-runtime') {
      path.remove()
    }
  }
})

const output = gen(ast, { jsescOption: { minimal: true }, retainLines: false, comments: true }, source)

writeFileSync(outPath, output.code)

console.log(`jsx: ${stats.jsx}  jsxs: ${stats.jsxs}  skipped: ${stats.skipped.length}`)

for (const skip of stats.skipped.slice(0, 40)) {
  console.log(`  skip ${skip}`)
}
