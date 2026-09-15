#!/usr/bin/env node
import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
const SOURCE_EXTENSIONS = new Set(['.ts', '.tsx', '.css'])
const TEST_FILE = /\.(?:test|spec)\.[^.]+$/
const RAW_COLOR = /#[0-9a-fA-F]{3,8}(?![0-9a-fA-F])/g

// These files define themes, terminal palettes, or third-party brand artwork.
// They are deliberately allowed to retain authored colors until their owning
// systems expose equivalent design tokens. New component styles are not allowed
// to use this escape hatch.
const ALLOWED_PATHS = new Set([
  'app/right-sidebar/terminal/selection.ts',
  'app/messaging/platform-icon.tsx',
  'components/chat/code-editor-theme.ts',
  'components/onboarding-chat/options.tsx',
  'lib/mcp-brands.tsx',
  'lib/tour/app-tour.css',
  'plugins/kanban/kanban.css',
  'plugins/kanban/types.ts',
  'styles.css'
])

// Existing non-theme colors are an exact baseline: new colors still fail.
const SANCTIONED_LITERALS = new Map(Object.entries(JSON.parse('{"app/chat/composer/rich-editor.ts":["#039"],"app/chat/index.tsx":["#89843"],"app/chat/session-tile.tsx":["#f87171","#72245"],"app/quick-entry/quick-entry-app.tsx":["#8a8a8a","#eee"],"app/right-sidebar/review/churn-bar.tsx":["#000"],"app/right-sidebar/terminal/use-agent-terminal.ts":["#ffffff"],"app/right-sidebar/terminal/use-terminal-session.ts":["#ffffff"],"app/settings/billing/tier-art.tsx":["#0000f2"],"app/settings/gateway-settings.tsx":["#95942"],"app/skills/plugins-tab.tsx":["#f87171"],"app/starmap/color.ts":["#888888","#000","#fff"],"app/starmap/star-map.tsx":["#000"],"app/updates-overlay.tsx":["#45205"],"components/assistant-ui/markdown-text.tsx":["#000"],"components/chat/shiki-config.ts":["#6e7781","#57606a"],"components/chat/vibe-hearts.tsx":["#ff9ec4"],"components/intro-reveal/intro-reveal-surface.tsx":["#16171b","#0a0b0e"],"components/intro-reveal/scenes/style.ts":["#4d8dff"],"components/intro-reveal/viewport-cube.ts":["#ff0000","#00ff00","#0000ff"],"components/pane-shell/tree/zones-engine.ts":["#F5FCFF","#FFFFFF","#008CFF","#000000"],"components/pet/pet-star-shower.tsx":["#ffd76a","#9aa0ff","#fff","#ffffff"],"components/ui/pane-tab.tsx":["#000"],"contrib/runtime-loader.ts":["#66899"],"debug/perf-live.ts":["#c0392b","#27ae60","#fff","#5a7db0"],"lib/preview-act/watch-in-page.ts":["#EC008C","#FFFFFF","#111111"],"lib/preview-annotate/in-page.ts":["#2F80ED","#fff"],"lib/preview-annotate/tokens.ts":["#2F80ED","#2A2A2A","#F4F4F4","#3D3D3D"],"plugins/hermes-bots/avatar.tsx":["#8b5cf6"],"plugins/hermes-bots/group-chat-view.tsx":["#89788"],"plugins/hermes-bots/mcp-setup.tsx":["#f87171"],"plugins/hermes-bots/model-picker.tsx":["#95279"]}')))

function isAllowed(relativePath) {
  return relativePath.startsWith('themes/') || ALLOWED_PATHS.has(relativePath)
}

export function withoutComments(source) {
  let result = ''
  let index = 0
  let quote = null

  while (index < source.length) {
    const char = source[index]
    const next = source[index + 1]

    if (quote) {
      result += char
      if (char === '\\') {
        result += next ?? ''
        index += 2
        continue
      }
      if (char === quote) quote = null
      index += 1
      continue
    }

    if (char === '"' || char === "'" || char === '`') {
      quote = char
      result += char
      index += 1
      continue
    }
    if (char === '/' && next === '/') {
      const end = source.indexOf('\n', index)
      const comment = source.slice(index, end === -1 ? source.length : end)
      result += comment.replace(/[^\n]/g, ' ')
      index = end === -1 ? source.length : end
      continue
    }
    if (char === '/' && next === '*') {
      const end = source.indexOf('*/', index + 2)
      const comment = source.slice(index, end === -1 ? source.length : end + 2)
      result += comment.replace(/[^\n]/g, ' ')
      index = end === -1 ? source.length : end + 2
      continue
    }
    // A quote inside a regular expression is not the start of a string. Scan
    // regex bodies before the string branch so a pattern cannot accidentally
    // hide comments that follow it.
    if (char === '/' && next !== '/' && next !== '*') {
      const previous = source.slice(0, index).trimEnd()
      const preceding = previous.at(-1)
      const startsAfterKeyword = /\b(?:return|case|throw|yield|await)$/.test(previous)
      if (!preceding || /[=(:,[!&|?{;}]/.test(preceding) || startsAfterKeyword) {
        let end = index + 1
        let inCharacterClass = false
        while (end < source.length) {
          if (source[end] === '\\') end += 2
          else if (source[end] === '[') { inCharacterClass = true; end += 1 }
          else if (source[end] === ']') { inCharacterClass = false; end += 1 }
          else if (source[end] === '/' && !inCharacterClass) { end += 1; break }
          else end += 1
        }
        result += source.slice(index, end)
        index = end
        continue
      }
    }

    result += char
    index += 1
  }

  return result
}

async function sourceFiles(directory, root = directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    if (entry.name === 'node_modules' || entry.name === 'dist') continue
    const fullPath = path.join(directory, entry.name)
    if (entry.isDirectory()) files.push(...await sourceFiles(fullPath, root))
    else if (SOURCE_EXTENSIONS.has(path.extname(entry.name)) && !TEST_FILE.test(entry.name)) {
      files.push(path.relative(root, fullPath))
    }
  }
  return files
}

export async function findRawColors(sourceRoot) {
  const violations = []
  for (const relativePath of await sourceFiles(sourceRoot)) {
    if (isAllowed(relativePath)) continue
    const source = withoutComments(await readFile(path.join(sourceRoot, relativePath), 'utf8'))
    for (const match of source.matchAll(RAW_COLOR)) {
      if (SANCTIONED_LITERALS.get(relativePath)?.includes(match[0])) continue
      const line = source.slice(0, match.index).split('\n').length
      violations.push(`${relativePath}:${line}: ${match[0]}`)
    }
  }
  return violations
}

export async function main({ sourceRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'src') } = {}) {
  const violations = await findRawColors(sourceRoot)
  if (violations.length === 0) {
    console.log('Design token check passed: no raw hex colors outside sanctioned sources.')
    return 0
  }

  console.error('Raw hex colors are only allowed in documented theme, terminal, and brand sources:')
  for (const violation of violations) console.error(`  ${violation}`)
  return 1
}

if (process.argv[1] === fileURLToPath(import.meta.url)) process.exitCode = await main()
