// Build-time guard: refuse a module that both imports a name and declares it at
// the same (module) scope.
//
// This is a parse error — Babel answers `Duplicate declaration "X"` at
// `Scope.checkBlockScopedCollisions` — but the way it arrives is a MERGE
// RESOLUTION, not a typo. A refactor extracts a function into a sibling module
// and adds the import; a later catch-up merge resolves the same hunk by keeping
// the import *and* resurrecting the local copy.
//
// The local copy is not necessarily a stale duplicate: it can be the branch that
// still carries work the extracted module never received (a field added on one
// line of history but not the other). So "delete the duplicate" is not a safe
// repair — it builds green and silently drops behaviour. Catching it before the
// build costs one pass over the app's source and turns a failure deep inside the
// bundler into a line that names the merge and says the copies must be
// reconciled, not one of them deleted.

import { readdirSync, readFileSync } from "fs"
import { join, relative, resolve } from "path"
import { isMain } from "./utils.mjs"

const SOURCE_EXTENSIONS = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"])

// `import Default, { a, b as c } from '...'` / `import * as ns from '...'`.
// Type-only imports are skipped: `import type { Foo }` plus a value declaration
// of Foo is legal and common.
const IMPORT_RE =
  /^import\s+(?!type\b)(?:(?<default>[\w$]+)\s*,?\s*)?(?:\*\s+as\s+(?<namespace>[\w$]+)|\{(?<named>[^}]*)\})?\s*from\s*['"][^'"]+['"]/gm

// Module-scope declarations only (column 0). An inner function that shadows an
// import is legal, and flagging it would make the guard noise.
const DECLARATION_RE =
  /^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?(?:function\*?|class|const|let|var|enum)\s+([\w$]+)/gm

function localNames(namedList) {
  if (!namedList) return []
  return namedList
    .split(",")
    .map(entry => entry.trim())
    .filter(entry => entry && !entry.startsWith("type "))
    .map(entry => {
      const parts = entry.split(/\s+as\s+/)
      return (parts[1] ?? parts[0]).trim()
    })
    .filter(Boolean)
}

// Pure per-file check — returns [{name, importLine, declarationLine}].
export function shadowedImports(source) {
  const imported = new Map()
  for (const match of source.matchAll(IMPORT_RE)) {
    const line = source.slice(0, match.index).split("\n").length
    const names = [
      ...(match.groups?.default ? [match.groups.default] : []),
      ...(match.groups?.namespace ? [match.groups.namespace] : []),
      ...localNames(match.groups?.named)
    ]
    for (const name of names) if (!imported.has(name)) imported.set(name, line)
  }
  if (imported.size === 0) return []

  const conflicts = []
  for (const match of source.matchAll(DECLARATION_RE)) {
    const name = match[1]
    if (!imported.has(name)) continue
    const declarationLine = source.slice(0, match.index).split("\n").length
    conflicts.push({ name, importLine: imported.get(name), declarationLine })
  }
  return conflicts
}

function sourceFiles(dir) {
  const files = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "node_modules" || entry.name === "dist" || entry.name === "build") continue
    const full = join(dir, entry.name)
    if (entry.isDirectory()) files.push(...sourceFiles(full))
    else if (SOURCE_EXTENSIONS.has(entry.name.slice(entry.name.lastIndexOf("."))) && !entry.name.endsWith(".d.ts")) {
      files.push(full)
    }
  }
  return files
}

// Pure check over a tree — returns { ok: true } or { ok: false, violations: [...] }.
export function checkShadowedImports(srcDir) {
  const violations = []
  for (const file of sourceFiles(srcDir)) {
    for (const conflict of shadowedImports(readFileSync(file, "utf8"))) {
      violations.push({ file, ...conflict })
    }
  }
  return violations.length === 0 ? { ok: true } : { ok: false, violations }
}

function main() {
  const app = resolve(import.meta.dirname, "..")
  const srcDir = join(app, "src")
  const result = checkShadowedImports(srcDir)
  if (result.ok) return

  const lines = result.violations.map(
    v => `  ${relative(app, v.file)}: '${v.name}' imported at line ${v.importLine} and declared at line ${v.declarationLine}`
  )
  console.error(
    "✗ assert-no-shadowed-imports: a name is both imported and declared at module scope.\n" +
      `${lines.join("\n")}\n` +
      "That is a parse error the bundler reports as a duplicate declaration. It normally arrives\n" +
      "from a merge resolution that kept an extraction's import and restored the local copy — the\n" +
      "two are not necessarily identical, so reconcile them (port anything the local copy still\n" +
      "does into the imported module) instead of deleting one."
  )
  process.exit(1)
}

if (isMain(import.meta.url)) {
  main()
}
