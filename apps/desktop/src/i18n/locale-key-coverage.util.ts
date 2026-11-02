import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import ts from 'typescript'

/**
 * Shared extraction logic for the locale key coverage guard
 * (locale-key-coverage.test.ts) and its baseline-regeneration script
 * (generate-locale-key-coverage-baseline.ts). Keep the two in sync.
 *
 * Parses locale catalog source files directly, before defineLocale()'s
 * merge, since the merged TRANSLATIONS export backfills every missing
 * key from en and always reports zero missing keys regardless of actual
 * coverage.
 *
 * The walker's grammar is intentionally closed: every ObjectLiteralExpression
 * member and every property-value shape it does not explicitly recognize
 * throws rather than silently falling back to "treat as one opaque leaf" or
 * "skip". A silent fallback is exactly how this guard's two known past bugs
 * happened (empty objects counted as leaves, defineFieldCopy/imported-const
 * blocks collapsed to one leaf, both hiding real coverage gaps) - throwing
 * turns the next unrecognized shape into a loud CI failure a maintainer has
 * to look at, instead of a silent under-count nobody notices.
 */

const CONSTANTS_PATH = join(import.meta.dirname, '..', 'app', 'settings', 'constants.ts')

/**
 * Property initializers that resolve to a FieldCopyTree defined in
 * src/app/settings/constants.ts rather than inline in the locale file
 * itself (en.ts: `fieldLabels: FIELD_LABELS`). This is a small, explicit
 * allowlist of the two known cross-file references - not general
 * identifier resolution, which a syntax-tree walk cannot do without a
 * full program/checker. Any other bare identifier value is unsupported
 * and fails the walk rather than being silently treated as an opaque leaf.
 */
const EXTERNAL_FIELD_COPY_NAMES = new Set(['FIELD_LABELS', 'FIELD_DESCRIPTIONS'])

function parseSource(filePath: string): ts.SourceFile {
  const source = readFileSync(filePath, 'utf8')

  return ts.createSourceFile(filePath, source, ts.ScriptTarget.Latest, true)
}

/** Peels `expr as T` down to the underlying expression, repeatedly. */
function peelAsExpression(expr: ts.Expression): ts.Expression {
  let current = expr

  while (ts.isAsExpression(current)) {
    current = current.expression
  }

  return current
}

/**
 * Finds the object literal a locale file's top-level exported const
 * resolves to - either a plain object (`en.ts`: `export const en = {...}`)
 * or the first argument of a call expression (every other locale:
 * `export const ar = defineLocale({...})`). Peels a top-level `as T`
 * first so a hypothetical `export const x = {...} as T` resolves too.
 */
function findExportedObjectLiteral(sourceFile: ts.SourceFile): ts.ObjectLiteralExpression | null {
  let found: ts.ObjectLiteralExpression | null = null

  ts.forEachChild(sourceFile, node => {
    if (!ts.isVariableStatement(node)) {
      return
    }

    for (const decl of node.declarationList.declarations) {
      const rawInit = decl.initializer

      if (!rawInit) {
        continue
      }

      const init = peelAsExpression(rawInit)

      if (ts.isObjectLiteralExpression(init)) {
        found = init
      } else if (ts.isCallExpression(init)) {
        const arg = init.arguments[0]

        if (arg && ts.isObjectLiteralExpression(arg)) {
          found = arg
        }
      }
    }
  })

  return found
}

function findNamedDeclarationInitializer(sourceFile: ts.SourceFile, name: string): ts.Expression | null {
  let found: ts.Expression | null = null

  ts.forEachChild(sourceFile, node => {
    if (!ts.isVariableStatement(node)) {
      return
    }

    for (const decl of node.declarationList.declarations) {
      if (ts.isIdentifier(decl.name) && decl.name.text === name && decl.initializer) {
        found = decl.initializer
      }
    }
  })

  return found
}

function isDefineFieldCopyCall(expr: ts.Expression): expr is ts.CallExpression {
  return ts.isCallExpression(expr) && ts.isIdentifier(expr.expression) && expr.expression.text === 'defineFieldCopy'
}

let constantsSourceFile: ts.SourceFile | null = null
const externalFieldCopyCache = new Map<string, ts.ObjectLiteralExpression | null>()

/**
 * Resolves one of EXTERNAL_FIELD_COPY_NAMES to the object literal argument
 * of its `defineFieldCopy({...})` definition in constants.ts. Both
 * FIELD_LABELS and FIELD_DESCRIPTIONS are themselves defineFieldCopy
 * calls there, so this reuses the same unwrap as an inline call.
 */
function resolveExternalFieldCopy(name: string): ts.ObjectLiteralExpression {
  if (externalFieldCopyCache.has(name)) {
    const cached = externalFieldCopyCache.get(name)!

    if (!cached) {
      throw new Error(`locale-key-coverage: ${name} did not resolve to a defineFieldCopy({...}) call in ${CONSTANTS_PATH}`)
    }

    return cached
  }

  if (!constantsSourceFile) {
    constantsSourceFile = parseSource(CONSTANTS_PATH)
  }

  const initializer = findNamedDeclarationInitializer(constantsSourceFile, name)
  let result: ts.ObjectLiteralExpression | null = null

  if (initializer && isDefineFieldCopyCall(initializer)) {
    const arg = initializer.arguments[0]

    if (arg && ts.isObjectLiteralExpression(arg)) {
      result = arg
    }
  }

  externalFieldCopyCache.set(name, result)

  if (!result) {
    throw new Error(`locale-key-coverage: ${name} did not resolve to a defineFieldCopy({...}) call in ${CONSTANTS_PATH}`)
  }

  return result
}

/** True for the scalar/callable leaf shapes every catalog currently uses. */
function isKnownLeafKind(expr: ts.Expression): boolean {
  return (
    ts.isStringLiteralLike(expr) ||
    ts.isArrowFunction(expr) ||
    ts.isFunctionExpression(expr) ||
    ts.isArrayLiteralExpression(expr) ||
    ts.isNumericLiteral(expr) ||
    expr.kind === ts.SyntaxKind.TrueKeyword ||
    expr.kind === ts.SyntaxKind.FalseKeyword
  )
}

/**
 * Dotted leaf-key paths in an object literal, depth-first. Every shape is
 * explicitly recognized; anything else throws (see module doc).
 *
 * - A non-empty nested object literal is descended into.
 * - An empty object literal (`platformIntro: {}`) contributes zero
 *   leaves - it is an extension point locales are meant to populate
 *   (see locale-parity.test.ts, #71233), not itself a translatable key.
 * - `defineFieldCopy({...})` calls are unwrapped and their argument is
 *   walked the same way, so a locale's inline call and en.ts's
 *   FIELD_LABELS / FIELD_DESCRIPTIONS reference compare at the same
 *   per-field granularity instead of collapsing to one opaque leaf.
 * - `expr as T` is peeled to `expr` before classification, so
 *   `{...} as Record<string, string>` behaves exactly like a bare
 *   object literal instead of falling back to an opaque leaf.
 * - String/template literals, arrow/function expressions, array
 *   literals, numeric literals, and boolean keywords are leaves.
 * - A dotted string-literal key (`'view.toggleHud': '...'`) is not
 *   split - its full text becomes one path segment, which happens to
 *   dot-join identically to a nested equivalent because '.' is also
 *   the segment separator this walker uses.
 */
export function collectKeyPaths(obj: ts.ObjectLiteralExpression, trail: string[] = []): string[] {
  const paths: string[] = []

  for (const prop of obj.properties) {
    if (!ts.isPropertyAssignment(prop)) {
      throw new Error(
        `locale-key-coverage: unsupported object member at ${trail.join('.') || '<root>'} ` +
          `(${ts.SyntaxKind[prop.kind]}) - only plain \`key: value\` properties are supported. ` +
          'Update collectKeyPaths if this is a legitimate new catalog shape.'
      )
    }

    const name = ts.isIdentifier(prop.name) || ts.isStringLiteral(prop.name) ? prop.name.text : null

    if (name === null) {
      throw new Error(
        `locale-key-coverage: unsupported property name at ${trail.join('.') || '<root>'} ` +
          `(${ts.SyntaxKind[prop.name.kind]}) - only identifier or string-literal keys are supported.`
      )
    }

    const path = [...trail, name]
    const init = peelAsExpression(prop.initializer)

    if (isDefineFieldCopyCall(init)) {
      const arg = init.arguments[0]

      if (!arg || !ts.isObjectLiteralExpression(arg)) {
        throw new Error(`locale-key-coverage: defineFieldCopy(...) at ${path.join('.')} does not take an object literal argument`)
      }

      paths.push(...collectKeyPaths(arg, path))

      continue
    }

    if (ts.isIdentifier(init)) {
      if (!EXTERNAL_FIELD_COPY_NAMES.has(init.text)) {
        throw new Error(
          `locale-key-coverage: unsupported identifier reference "${init.text}" at ${path.join('.')} - ` +
            `only ${[...EXTERNAL_FIELD_COPY_NAMES].join(', ')} are resolved from constants.ts. ` +
            'Add it to EXTERNAL_FIELD_COPY_NAMES if this is a legitimate new cross-file reference.'
        )
      }

      paths.push(...collectKeyPaths(resolveExternalFieldCopy(init.text), path))

      continue
    }

    if (ts.isObjectLiteralExpression(init)) {
      if (init.properties.length === 0) {
        continue
      }

      paths.push(...collectKeyPaths(init, path))

      continue
    }

    if (isKnownLeafKind(init)) {
      paths.push(path.join('.'))

      continue
    }

    throw new Error(
      `locale-key-coverage: unsupported value shape at ${path.join('.')} (${ts.SyntaxKind[init.kind]}) - ` +
        'update isKnownLeafKind (or add a dedicated unwrap) if this is a legitimate new catalog shape.'
    )
  }

  return paths
}

/** Dotted leaf-key paths defined by `locale`'s catalog source file. */
export function localeKeyPaths(locale: string): Set<string> {
  const filePath = join(import.meta.dirname, `${locale}.ts`)
  const sourceFile = parseSource(filePath)
  const obj = findExportedObjectLiteral(sourceFile)

  if (!obj) {
    throw new Error(`locale-key-coverage: could not find an exported object literal in ${filePath}`)
  }

  return new Set(collectKeyPaths(obj))
}

/** `en`'s leaf-key paths that `locale` does not define, sorted. */
export function missingKeyPaths(locale: string): string[] {
  const enKeys = localeKeyPaths('en')
  const localeKeys = localeKeyPaths(locale)

  return [...enKeys].filter(key => !localeKeys.has(key)).sort()
}

/**
 * Parses `source` (a self-contained snippet like
 * `export const x = { a: 'b' }`) and returns the object literal its
 * top-level export resolves to, using the same resolution
 * (defineFieldCopy/AsExpression-aware) as the real catalog files.
 * Exported for locale-key-coverage.test.ts's direct unit tests of
 * collectKeyPaths - not used by the guard itself.
 */
export function parseObjectLiteralForTest(source: string): ts.ObjectLiteralExpression {
  const sourceFile = ts.createSourceFile('test.ts', source, ts.ScriptTarget.Latest, true)
  const obj = findExportedObjectLiteral(sourceFile)

  if (!obj) {
    throw new Error('parseObjectLiteralForTest: no exported object literal found')
  }

  return obj
}
