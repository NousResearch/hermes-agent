/**
 * Prove the de-JSX codemod was lossless.
 *
 * Recompiles the generated .tsx back down to jsx-runtime calls with esbuild,
 * then compares it against the original hand-written source. Both sides are
 * normalized through the same printer so the diff reflects semantics, not
 * formatting.
 *
 * jsx/jsxs selection is deliberately collapsed: the pair differ only in React's
 * dev-mode static-children key warning, and esbuild picks between them on its
 * own rules rather than the ones the file was hand-written with.
 */
import { readFileSync } from 'node:fs'

import generate from '@babel/generator'
import { parse } from '@babel/parser'
import traverse from '@babel/traverse'
import * as t from '@babel/types'
import { transformSync } from 'esbuild'

const gen = generate.default ?? generate
const walk = traverse.default ?? traverse

const [, , originalPath, convertedPath] = process.argv

const allowed = { spreadChildren: 0, redundantKeyProp: 0 }

function normalize(code, plugins) {
  const ast = parse(code, { sourceType: 'module', plugins })

  walk(ast, {
    /**
     * JSX has no spread-children syntax: `children: [...items.map(f), x]` can
     * only be written `{items.map(f)}<X/>`, which compiles to
     * `children: [items.map(f), x]`. React flattens nested array children and
     * the mapped elements keep their explicit keys, so the two are equivalent
     * — this is the idiom the whole ecosystem writes. Fold the spread away on
     * the original side so the comparison doesn't flag it, and count it.
     */
    ObjectProperty(path) {
      if (!t.isIdentifier(path.node.key, { name: 'children' }) || !t.isArrayExpression(path.node.value)) {
        return
      }

      path.node.value.elements = path.node.value.elements.map(element => {
        if (!t.isSpreadElement(element)) {
          return element
        }

        allowed.spreadChildren += 1

        return element.argument
      })
    },
    // Drop captured raw text so both sides re-print literals canonically —
    // one quote style, and `48000` rather than esbuild's `48e3`.
    'StringLiteral|NumericLiteral'(path) {
      delete path.node.extra
    },
    // Template chunks keep their raw text, so an emoji written as a surrogate
    // pair on one side and a code point on the other reads as a diff.
    TemplateElement(path) {
      const { cooked } = path.node.value

      if (typeof cooked === 'string') {
        path.node.value.raw = cooked.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')
      }
    },
    /**
     * Some call sites pass `key` twice — inside props AND as the third
     * argument. React's jsx runtime takes the key from the third argument and
     * never copies `key` into props, so the props copy is dead either way.
     */
    CallExpression(path) {
      const callee = path.node.callee
      const [, props] = path.node.arguments

      if (!t.isIdentifier(callee) || callee.name !== 'jsx' || !t.isObjectExpression(props)) {
        return
      }

      props.properties = props.properties.filter(prop => {
        if (t.isObjectProperty(prop) && !prop.computed && t.isIdentifier(prop.key, { name: 'key' })) {
          allowed.redundantKeyProp += 1

          return false
        }

        return true
      })
    },
    // esbuild folds adjacent string/template concatenation into one literal.
    BinaryExpression: {
      exit(path) {
        const { operator, left, right } = path.node

        if (operator !== '+') {
          return
        }

        if (t.isStringLiteral(left) && t.isStringLiteral(right)) {
          path.replaceWith(t.stringLiteral(left.value + right.value))

          return
        }

        // `` `a${x}b` + 'c' `` → `` `a${x}bc` ``
        if (t.isTemplateLiteral(left) && t.isStringLiteral(right)) {
          const quasis = left.quasis.map(quasi => t.cloneNode(quasi))
          const last = quasis[quasis.length - 1]

          last.value = { raw: last.value.cooked + right.value, cooked: last.value.cooked + right.value }
          path.replaceWith(t.templateLiteral(quasis, left.expressions))

          return
        }

        // `'a' + `b${x}` ` → `` `ab${x}` ``
        if (t.isStringLiteral(left) && t.isTemplateLiteral(right)) {
          const quasis = right.quasis.map(quasi => t.cloneNode(quasi))
          const first = quasis[0]

          first.value = { raw: left.value + first.value.cooked, cooked: left.value + first.value.cooked }
          path.replaceWith(t.templateLiteral(quasis, right.expressions))

          return
        }

        // `` `a${x}` + `b${y}` `` → `` `a${x}b${y}` `` — the seam quasis merge.
        if (t.isTemplateLiteral(left) && t.isTemplateLiteral(right)) {
          const leftQuasis = left.quasis.map(quasi => t.cloneNode(quasi))
          const rightQuasis = right.quasis.map(quasi => t.cloneNode(quasi))
          const seam = leftQuasis.pop()
          const head = rightQuasis.shift()
          const merged = (seam.value.cooked ?? '') + (head.value.cooked ?? '')

          seam.value = { raw: merged, cooked: merged }
          path.replaceWith(
            t.templateLiteral([...leftQuasis, seam, ...rightQuasis], [...left.expressions, ...right.expressions])
          )
        }
      }
    },
    // esbuild prefers a template literal when it saves escaping.
    TemplateLiteral: {
      exit(path) {
        if (path.node.expressions.length === 0) {
          path.replaceWith(t.stringLiteral(path.node.quasis[0].value.cooked ?? ''))
        }
      }
    },
    // esbuild rewrites `undefined` to `void 0`.
    UnaryExpression(path) {
      if (path.node.operator === 'void' && t.isNumericLiteral(path.node.argument, { value: 0 })) {
        path.replaceWith(t.identifier('undefined'))
      }
    },
    // esbuild emits `import { jsx as _jsx }`; the original imports it bare.
    Identifier(path) {
      if (path.node.name === '_jsx' || path.node.name === '_jsxs') {
        path.node.name = 'jsx'
      }

      if (path.node.name === 'jsxs') {
        path.node.name = 'jsx'
      }
    },
    ImportDeclaration(path) {
      if (path.node.source.value === 'react/jsx-runtime') {
        path.remove()
      }
    },
    // esbuild hoists the spread helper for `{...props}` on intrinsics.
    VariableDeclarator(path) {
      if (t.isIdentifier(path.node.id) && /^__(spread|assign|objRest)/.test(path.node.id.name)) {
        path.remove()
      }
    }
  })

  // esbuild hoists `export default {…}` into
  // `var X = {…}; export { X as default }`.
  const body = ast.program.body
  const exportIndex = body.findIndex(
    node =>
      (t.isExportDefaultDeclaration(node) && t.isIdentifier(node.declaration)) ||
      (t.isExportNamedDeclaration(node) &&
        !node.declaration &&
        node.specifiers.length === 1 &&
        t.isExportSpecifier(node.specifiers[0]) &&
        t.isIdentifier(node.specifiers[0].exported, { name: 'default' }))
  )

  if (exportIndex !== -1) {
    const exported = body[exportIndex]
    const name = t.isExportDefaultDeclaration(exported)
      ? exported.declaration.name
      : exported.specifiers[0].local.name
    const declIndex = body.findIndex(
      node =>
        t.isVariableDeclaration(node) &&
        node.declarations.length === 1 &&
        t.isIdentifier(node.declarations[0].id, { name })
    )

    if (declIndex !== -1) {
      body[exportIndex] = t.exportDefaultDeclaration(body[declIndex].declarations[0].init)
      body.splice(declIndex, 1)
    }
  }

  // esbuild renames locals that shadow an outer binding (`displayName` →
  // `displayName2`). Canonically rename every binding in declaration order on
  // both sides so the comparison tests alpha-equivalence rather than spelling.
  // Numbered per scope, not globally: one extra binding on one side then
  // shifts only its own scope instead of desynchronizing the whole file.
  let scopeId = 0

  walk(ast, {
    Scopable(path) {
      const bindings = Object.values(path.scope.bindings).sort(
        (a, b) => (a.identifier.start ?? 0) - (b.identifier.start ?? 0)
      )
      const scope = (scopeId += 1)

      bindings.forEach((binding, index) => {
        path.scope.rename(binding.identifier.name, `__s${scope}_${index}`)
      })
    }
  })

  return gen(ast, { comments: false, compact: true, jsescOption: { minimal: true } }).code
}

const original = normalize(readFileSync(originalPath, 'utf8'), [])
const recompiled = normalize(
  transformSync(readFileSync(convertedPath, 'utf8'), {
    loader: 'tsx',
    jsx: 'automatic',
    format: 'esm',
    target: 'esnext'
  }).code,
  []
)

console.log(
  `allowed rewrites — spread children folded: ${allowed.spreadChildren}, redundant key props dropped: ${allowed.redundantKeyProp}`
)

if (original === recompiled) {
  console.log(`IDENTICAL — ${original.length} chars of normalized output match`)
  process.exit(0)
}

console.log(`DIFFER — original ${original.length} chars, recompiled ${recompiled.length} chars`)

// Report the first divergence with surrounding context so it can be chased.
let i = 0
while (i < original.length && original[i] === recompiled[i]) {
  i += 1
}

const window = 260
console.log(`\nfirst divergence at char ${i}:`)
console.log(`\n--- original ---\n${original.slice(Math.max(0, i - window), i + window)}`)
console.log(`\n--- recompiled ---\n${recompiled.slice(Math.max(0, i - window), i + window)}`)
process.exit(1)
