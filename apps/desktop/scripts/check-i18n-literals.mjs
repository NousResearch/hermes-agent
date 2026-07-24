import fs from 'node:fs'
import path from 'node:path'

import ts from 'typescript'

const SOURCE_ROOT = path.resolve('src')
const USER_FACING_ATTRIBUTES = new Set(['alt', 'aria-label', 'label', 'placeholder', 'text', 'title'])
const SOURCE_FILE = /\.tsx$/
const EXCLUDED_FILE = /(?:\.test|\.stories)\.tsx$/

// Intentional product names, syntax examples, paths, and accessibility-only
// component identifiers. Everything else must flow through the i18n catalog.
const ALLOWED_LITERALS = [
  /^\/help$/,
  /^@(?:file|folder):$/,
  /^https?:\/\//,
  /^https:\/\/…$/,
  /^· SOUL\.md$/,
  /^SOUL\.md$/,
  /^\.env$/,
  /^mcp\.json$/,
  /^~\/\.local\/bin$/,
  /^%LOCALAPPDATA%\\Hermes\\bin$/,
  /^%LOCALAPPDATA%\\hermes\\logs\\$/,
  /^\/SKILL\.md$/,
  /^(?:Nous Portal|OpenRouter|OpenAI|Fireworks AI)$/,
  /^HERMES$/,
  /^(?:OAuth|API key|Esc|MoA:)$/,
  /^(?:my-profile|file-tree|pet-overlay|root)$/
]

function listFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const target = path.join(directory, entry.name)

    if (entry.isDirectory()) {
      return listFiles(target)
    }

    return SOURCE_FILE.test(target) && !EXCLUDED_FILE.test(target) ? [target] : []
  })
}

function normalized(value) {
  return value.replace(/\s+/g, ' ').trim()
}

function isUserFacingLiteral(value) {
  return /[A-Za-z]{3}/.test(value) && !ALLOWED_LITERALS.some(pattern => pattern.test(value))
}

const violations = []

for (const file of listFiles(SOURCE_ROOT)) {
  const sourceText = fs.readFileSync(file, 'utf8')
  const sourceFile = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)

  const check = (node, rawValue) => {
    const value = normalized(rawValue)

    if (!value || !isUserFacingLiteral(value)) {
      return
    }

    const { line, character } = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile))
    violations.push(`${path.relative(process.cwd(), file)}:${line + 1}:${character + 1}  ${JSON.stringify(value)}`)
  }

  const visit = node => {
    if (ts.isJsxText(node)) {
      check(node, node.text)
    } else if (
      ts.isJsxAttribute(node) &&
      USER_FACING_ATTRIBUTES.has(node.name.text) &&
      node.initializer &&
      ts.isStringLiteral(node.initializer)
    ) {
      check(node, node.initializer.text)
    }

    ts.forEachChild(node, visit)
  }

  visit(sourceFile)
}

if (violations.length) {
  console.error('User-facing English literals must use the desktop i18n catalog:')
  console.error(violations.join('\n'))
  process.exitCode = 1
} else {
  console.log('Desktop i18n literal check passed.')
}
