import { readFile, writeFile } from 'node:fs/promises'
import { spawn } from 'node:child_process'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { workspaceTool } from './build/frontend-common.mjs'

const root = process.cwd()
const localeDir = path.join(root, 'locales')
const webEnglishPath = path.join(root, 'web/src/i18n/en.ts')
// Python interpreter running the local Argos/ LibreTranslate engine. Override with
// HERMES_PYTHON when argostranslate lives in a dedicated environment.
const pythonExecutable = process.env.HERMES_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')

// Hermes home, resolved with the same precedence as the Python side (HERMES_HOME
// first, then the platform default) so the model cache paths stay portable.
const hermesHome = process.env.HERMES_HOME || (process.platform === 'win32'
  ? path.join(process.env.LOCALAPPDATA || path.join(process.env.USERPROFILE || '', 'AppData', 'Local'), 'hermes')
  : path.join(process.env.XDG_DATA_HOME || path.join(process.env.HOME || '', '.local', 'share'), 'hermes'))
const cacheRoot = path.join(hermesHome, 'cache')
const translateScript = path.join(root, 'scripts/translate_libre.py')
const batchCharacterLimit = 2400
const concurrency = 4

const keepTerms = [
  'Hermes', 'Nous Research', 'OpenAI', 'Anthropic', 'Claude', 'Telegram', 'Discord', 'Slack',
  'Signal', 'WhatsApp', 'Matrix', 'MCP', 'Piper', 'Whisper', 'Groq', 'GitHub', 'GitLab',
  'Gemini', 'OpenRouter', 'Ollama', 'DeepSeek', 'Qwen', 'Mistral', 'Kimi', 'Tavily', 'Brave',
  'Exa', 'Firecrawl', 'Jina', 'ElevenLabs', 'TUI', 'GUI', 'CLI', 'API',
]
const manualTranslations = new Map([
  ['ON', 'PORNIT'], ['OFF', 'OPRIT'], ['On', 'Pornit'], ['Off', 'Oprit'], ['on', 'pornit'], ['off', 'oprit'],
  ['Back', 'Înapoi'], ['Cancel', 'Anulează'], ['Close', 'Închide'], ['Confirm', 'Confirmă'],
  ['Continue', 'Continuă'], ['Copy', 'Copiază'], ['Copied', 'Copiat'], ['Delete', 'Șterge'],
  ['Done', 'Finalizat'], ['Edit', 'Editează'], ['Language', 'Limbă'], ['Loading…', 'Se încarcă…'],
  ['Loading...', 'Se încarcă...'], ['Message', 'Mesaj'], ['New chat', 'Conversație nouă'],
  ['No', 'Nu'], ['Open', 'Deschide'], ['Retry', 'Reîncearcă'], ['Save', 'Salvează'],
  ['Search', 'Caută'], ['Send', 'Trimite'], ['Send a message', 'Trimite un mesaj'],
  ['Settings', 'Setări'], ['Submit', 'Trimite'], ['Yes', 'Da'], ['Approve', 'Aprobă'],
  ['Reject', 'Respinge'], ['Allow', 'Permite'], ['Always allow', 'Permite întotdeauna'],
  ['Never allow', 'Nu permite niciodată'], ['Romanian', 'Română'], ['Choose a language', 'Alege o limbă'],
  ['Type a message...', 'Scrie un mesaj...'], ['Type a message…', 'Scrie un mesaj…'],
  ['Start a new chat', 'Începe o conversație nouă'], ['Start a conversation', 'Începe o conversație'],
  ['What can I help you with?', 'Cu ce te pot ajuta?'], ['Are you sure?', 'Sigur?'],
  ['Running', 'Rulează'], ['running', 'rulează'], ['Not running', 'Nu rulează'],
  ['Logs', 'Jurnale'], ['logs', 'jurnale'], ['Tokens', 'Tokeni'], ['tokens', 'tokeni'],
  ['Slug', 'Slug'], ['slug', 'slug'], ['Todo', 'De făcut'], ['Ready', 'Pregătit'],
])

const yamlPath = workspaceTool(root, 'apps/desktop', 'yaml')
const yaml = await import(pathToFileURL(yamlPath).href)
const esbuildPath = workspaceTool(root, 'apps/desktop', 'esbuild')
const { build } = await import(pathToFileURL(esbuildPath).href)
const typescriptPath = workspaceTool(root, 'apps/desktop', 'typescript')
const tsModule = await import(pathToFileURL(typescriptPath).href)
const ts = tsModule.default ?? tsModule

const tokenPattern = /https?:\/\/[^\s<>]+|`[^`]*`|<[^>]+>|\x1b\[[0-?]*[ -/]*[@-~]|\{[^{}]*\}|%(?:\d+\$)?[a-zA-Z]|(?<![\w`])\/[A-Za-z][A-Za-z0-9_-]*/g
const placeholderPattern = /\{(?:[A-Za-z_][\w.-]*|\d+)\}/g

function protect(text, phraseIndex) {
  const tokens = []
  let protectedText = text.replace(tokenPattern, (value) => {
    const token = `ZXQKEEP${phraseIndex.toString(36)}X${tokens.length.toString(36)}QXZ`
    tokens.push([token, value])
    return token
  })
  for (const [index, term] of keepTerms.entries()) {
    const expression = new RegExp(`(?<![\\p{L}\\p{N}_])${term.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}(?![\\p{L}\\p{N}_])`, 'giu')
    protectedText = protectedText.replace(expression, (value) => {
      const token = `ZXQBRAND${phraseIndex.toString(36)}X${index.toString(36)}QXZ`
      tokens.push([token, value])
      return token
    })
  }
  return { protectedText, tokens }
}

function restore(text, tokens) {
  let restored = text
  for (const [token, original] of tokens) {
    restored = restored.replaceAll(token, original)
  }
  return restored
}

function getPlaceholders(text) {
  return [...text.matchAll(placeholderPattern)].map((match) => match[0]).sort()
}

function isWorthTranslating(text) {
  return /[A-Za-z]/.test(text) && !/^\s*$/.test(text)
}

function manualTranslation(text) {
  const leading = text.match(/^\s*/)?.[0] ?? ''
  const trailing = text.match(/\s*$/)?.[0] ?? ''
  const trimmedEnd = trailing.length ? text.length - trailing.length : text.length
  const trimmed = text.slice(leading.length, trimmedEnd)
  const translation = manualTranslations.get(trimmed)
  return translation === undefined ? null : `${leading}${translation}${trailing}`
}

async function requestTranslation(text) {
  // 1. Try LibreTranslate local HTTP server if active
  try {
    const response = await fetch('http://127.0.0.1:5000/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: text, source: 'en', target: 'ro' }),
      signal: AbortSignal.timeout(2000),
    })
    if (response.ok) {
      const payload = await response.json()
      if (typeof payload?.translatedText === 'string') {
        return payload.translatedText
      }
    }
  } catch {
    // Fall back to direct local LibreTranslate / ArgosTranslate Python engine
  }

  // 2. Local LibreTranslate / ArgosTranslate engine via Python subprocess
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonExecutable, [translateScript], {
      env: {
        ...process.env,
        XDG_DATA_HOME: path.join(cacheRoot, 'data'),
        XDG_CONFIG_HOME: path.join(cacheRoot, 'config'),
        XDG_CACHE_HOME: path.join(cacheRoot, 'cache'),
      },
    })
    let stdout = ''
    let stderr = ''
    proc.stdout.on('data', (data) => { stdout += data })
    proc.stderr.on('data', (data) => { stderr += data })
    proc.on('close', (code) => {
      if (code === 0) resolve(stdout)
      else reject(new Error(`LibreTranslate engine failed (exit code ${code}): ${stderr}`))
    })
    proc.on('error', reject)
    proc.stdin.write(text)
    proc.stdin.end()
  })
}

async function translateBatch(values, firstIndex) {
  const entries = values.map((value, index) => {
    const phraseIndex = firstIndex + index
    const marker = `ZXQSEG${phraseIndex.toString(36)}XQXZ`
    const { protectedText, tokens } = protect(value, phraseIndex)
    return { value, marker, protectedText, tokens }
  })
  const combined = entries.map(({ marker, protectedText }) => `${marker}\n${protectedText}`).join('\n')
  try {
    const translated = await requestTranslation(combined)
    const results = entries.map((entry, index) => {
      const start = translated.indexOf(entry.marker)
      const next = index + 1 < entries.length ? translated.indexOf(entries[index + 1].marker, start + entry.marker.length) : translated.length
      if (start < 0 || next < 0) return null
      const candidate = restore(translated.slice(start + entry.marker.length, next).trim(), entry.tokens)
      return getPlaceholders(candidate).join('\u0000') === getPlaceholders(entry.value).join('\u0000') ? candidate : entry.value
    })
    if (results.every((result) => result !== null)) return results
  } catch (error) {
    console.warn(`Batch beginning at ${firstIndex} failed; retrying individually: ${error.message}`)
  }

  return Promise.all(entries.map(async (entry) => {
    try {
      const translated = await requestTranslation(entry.protectedText)
      const candidate = restore(translated.trim(), entry.tokens)
      return getPlaceholders(candidate).join('\u0000') === getPlaceholders(entry.value).join('\u0000') ? candidate : entry.value
    } catch (error) {
      console.warn(`Leaving one source string untranslated: ${error.message}`)
      return entry.value
    }
  }))
}

async function translateAll(strings, cachedTranslations = new Map()) {
  const unique = [...new Set(strings)].filter(isWorthTranslating)
  const translations = new Map(strings.map((value) => [value, value]))
  const pending = unique.filter((value) => manualTranslation(value) === null && !cachedTranslations.has(value))
  for (const value of unique) {
    const manual = manualTranslation(value)
    if (manual !== null) translations.set(value, manual)
    else if (cachedTranslations.has(value)) translations.set(value, cachedTranslations.get(value))
  }
  const batches = []
  let current = []
  let currentSize = 0
  for (const value of pending) {
    const itemSize = value.length + 36
    if (current.length && currentSize + itemSize > batchCharacterLimit) {
      batches.push(current)
      current = []
      currentSize = 0
    }
    current.push(value)
    currentSize += itemSize
  }
  if (current.length) batches.push(current)

  let nextBatch = 0
  const workers = Array.from({ length: Math.min(concurrency, batches.length) }, async () => {
    while (nextBatch < batches.length) {
      const batchIndex = nextBatch++
      const batch = batches[batchIndex]
      const results = await translateBatch(batch, batchIndex * 10000)
      batch.forEach((value, index) => translations.set(value, results[index]))
      if ((batchIndex + 1) % 25 === 0 || batchIndex + 1 === batches.length) {
        console.log(`Translated batch ${batchIndex + 1}/${batches.length}`)
      }
    }
  })
  await Promise.all(workers)
  return translations
}

function flatten(value, prefix = '', result = {}, branching = [], preserveArrays = false) {
  if (typeof value === 'function') {
    try {
      const args = Array.from({ length: value.length }, (_, index) => `{${index}}`)
      const rendered = String(value(...args))
      result[prefix] = rendered
      if (value.length > 0) {
        try {
          const one = String(value(...args.map((arg, index) => index === 0 ? 1 : arg)))
          const two = String(value(...args.map((arg, index) => index === 0 ? 2 : arg)))
          if (one.replace('1', 'N') !== two.replace('2', 'N')) branching.push(prefix)
        } catch {
          // Non-numeric first argument; the positional template above is still usable.
        }
      }
    } catch (error) {
      console.warn(`Could not render English desktop function ${prefix}: ${error.message}`)
    }
  } else if (typeof value === 'string') {
    result[prefix] = value
  } else if (Array.isArray(value)) {
    if (preserveArrays) {
      result[prefix] = value
    } else {
      value.forEach((item, index) => flatten(item, prefix ? `${prefix}.${index}` : String(index), result, branching, preserveArrays))
    }
  } else if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      flatten(child, prefix ? `${prefix}.${key}` : key, result, branching, preserveArrays)
    }
  }
  return result
}

function collectWebEdits(sourceFile, initializer) {
  const edits = []
  const add = (node, value, kind, pathParts) => {
    edits.push({ start: node.getStart(sourceFile), end: node.getEnd(), value, kind, path: pathParts.join('.') })
  }
  const walk = (node, pathParts) => {
    if (ts.isObjectLiteralExpression(node)) {
      for (const property of node.properties) {
        if (ts.isPropertyAssignment(property)) {
          const name = ts.isIdentifier(property.name) || ts.isStringLiteral(property.name) ? property.name.text : ''
          walk(property.initializer, name ? [...pathParts, name] : pathParts)
        } else if (ts.isMethodDeclaration(property) || ts.isShorthandPropertyAssignment(property)) {
          for (const child of property.getChildren(sourceFile)) walk(child, pathParts)
        }
      }
      return
    }
    if (ts.isPropertyAccessExpression(node) || ts.isIdentifier(node) || ts.isTypeNode(node)) return
    if (ts.isStringLiteral(node)) {
      add(node, node.text, 'literal', pathParts)
      return
    }
    if (ts.isNoSubstitutionTemplateLiteral(node)) {
      add(node, node.text, 'literal', pathParts)
      return
    }
    if (ts.isTemplateExpression(node)) {
      const head = node.head
      addTemplateChunk(head, 'head', pathParts)
      for (const span of node.templateSpans) {
        walk(span.expression, pathParts)
        addTemplateChunk(span.literal, ts.isTemplateTail(span.literal) ? 'tail' : 'middle', pathParts)
      }
      return
    }
    ts.forEachChild(node, (child) => walk(child, pathParts))
  }
  const addTemplateChunk = (node, kind, pathParts) => {
    const start = node.getStart(sourceFile) + 1
    const end = node.getEnd() - (kind === 'tail' ? 1 : 2)
    if (end > start) add({ getStart: () => start, getEnd: () => end }, node.text, 'template', pathParts)
  }
  walk(initializer, [])
  return edits
}

function getDotted(source, dottedKey) {
  return dottedKey.split('.').reduce((value, part) => value && typeof value === 'object' ? value[part] : undefined, source)
}

function getEnglishYamlValue(source, flattened, key) {
  const nested = getDotted(source, key)
  return typeof nested === 'string' ? nested : flattened[key]
}

function translatedYamlValue(value, translations) {
  if (typeof value === 'string') return translations.get(value) ?? value
  if (Array.isArray(value)) return value.map((item) => translatedYamlValue(item, translations))
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, translatedYamlValue(child, translations)]))
  }
  return value
}

function applyWebEdits(source, edits, translations) {
  let output = source
  for (const edit of [...edits].sort((a, b) => b.start - a.start)) {
    const value = translations.get(edit.value) ?? edit.value
    const replacement = edit.kind === 'template'
      ? value.replaceAll('\\', '\\\\').replaceAll('`', '\\`').replaceAll('${', '\\${')
      : JSON.stringify(value)
    output = output.slice(0, edit.start) + replacement + output.slice(edit.end)
  }
  return output.replace(/export const en(?=\s*:)/, 'export const ro')
}

const enYamlPath = path.join(localeDir, 'en.yaml')
const englishYaml = yaml.parse(await readFile(enYamlPath, 'utf8'))
const tuiKeyData = JSON.parse(await readFile(path.join(localeDir, '_keys.tui.json'), 'utf8'))
const desktopKeyData = JSON.parse(await readFile(path.join(localeDir, '_keys.desktop.json'), 'utf8'))
const tuiKeys = Array.isArray(tuiKeyData) ? tuiKeyData : tuiKeyData.keys
const desktopKeys = Array.isArray(desktopKeyData) ? desktopKeyData : desktopKeyData.keys

const desktopEntry = path.join(root, 'apps/desktop/src/i18n/en.ts')
const desktopBuild = await build({
  entryPoints: [desktopEntry],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  target: 'node26',
  logLevel: 'silent',
})
const desktopModuleUrl = `data:text/javascript;base64,${Buffer.from(desktopBuild.outputFiles[0].contents).toString('base64')}`
const desktopEnglish = (await import(desktopModuleUrl)).en
const desktopEnglishValues = flatten(desktopEnglish, '', {}, [], true)

const shared = path.join(root, 'apps/shared/src')
const tuiSource = path.join(root, 'ui-tui')
const tuiAliases = {
  '@hermes/shared/ansi': path.join(shared, 'ansi.ts'),
  '@hermes/shared/billing': path.join(shared, 'billing-types.ts'),
  '@hermes/shared/color': path.join(shared, 'color.ts'),
  '@hermes/shared/gateway-text': path.join(shared, 'gateway-text.ts'),
  '@hermes/shared/i18n': path.join(shared, 'i18n.ts'),
  '@hermes/shared/translucency': path.join(shared, 'translucency.ts'),
  '@hermes/shared': shared,
  '@': path.join(tuiSource, 'src'),
}
const tuiBuild = await build({
  entryPoints: [path.join(tuiSource, 'src/i18n/en.ts')],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  target: 'node26',
  alias: tuiAliases,
  loader: { '.css': 'empty', '.svg': 'empty', '.png': 'empty', '.woff2': 'empty' },
  logLevel: 'silent',
})
const tuiModuleUrl = `data:text/javascript;base64,${Buffer.from(tuiBuild.outputFiles[0].contents).toString('base64')}`
const tuiEnglish = (await import(tuiModuleUrl)).en
const tuiBranching = []
const tuiEnglishValues = flatten(tuiEnglish, '', {}, tuiBranching)

const webEnglish = await readFile(webEnglishPath, 'utf8')
const webSourceFile = ts.createSourceFile(webEnglishPath, webEnglish, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
const enDeclaration = webSourceFile.statements
  .filter(ts.isVariableStatement)
  .flatMap((statement) => [...statement.declarationList.declarations])
  .find((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === 'en')
if (!enDeclaration?.initializer) throw new Error('Could not find the web English catalog export')
const webEdits = collectWebEdits(webSourceFile, enDeclaration.initializer)
const webEnglishValues = webEdits.map((edit) => edit.value)

const englishYamlValues = flatten(englishYaml)
const cachedTranslations = new Map()
function seedCachedTranslations(englishValues, romanianValues) {
  for (const [key, source] of Object.entries(englishValues)) {
    const translated = romanianValues[key]
    if (typeof source === 'string' && typeof translated === 'string' && translated !== key) {
      cachedTranslations.set(source, translated)
    }
  }
}
seedCachedTranslations(englishYamlValues, flatten(yaml.parse(await readFile(path.join(localeDir, 'ro.yaml'), 'utf8'))))
seedCachedTranslations(tuiEnglishValues, yaml.parse(await readFile(path.join(localeDir, 'ro.tui.yaml'), 'utf8')))
seedCachedTranslations(desktopEnglishValues, yaml.parse(await readFile(path.join(localeDir, 'ro.desktop.yaml'), 'utf8')))
const cachedWebPath = path.join(root, 'web/src/i18n/ro.ts')
const cachedWebSource = await readFile(cachedWebPath, 'utf8')
const cachedWebSourceFile = ts.createSourceFile(cachedWebPath, cachedWebSource, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
const roDeclaration = cachedWebSourceFile.statements
  .filter(ts.isVariableStatement)
  .flatMap((statement) => [...statement.declarationList.declarations])
  .find((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === 'ro')
if (roDeclaration?.initializer) {
  const cachedWebValues = new Map(collectWebEdits(cachedWebSourceFile, roDeclaration.initializer).map((edit) => [edit.path, edit.value]))
  for (const edit of webEdits) {
    const translated = cachedWebValues.get(edit.path)
    if (typeof translated === 'string' && translated !== edit.path) cachedTranslations.set(edit.value, translated)
  }
}
const allSourceStrings = [
  ...Object.values(englishYamlValues),
  ...tuiKeys.map((key) => tuiEnglishValues[key]).filter((value) => typeof value === 'string'),
  ...desktopKeys.flatMap((key) => {
    const value = desktopEnglishValues[key]
    return typeof value === 'string' ? [value] : Array.isArray(value) ? value.filter((item) => typeof item === 'string') : []
  }),
  ...webEnglishValues,
]
console.log(`Translating ${new Set(allSourceStrings).size} unique English strings.`)
const translations = await translateAll(allSourceStrings, cachedTranslations)

const romanianYaml = translatedYamlValue(englishYaml, translations)
const missingTuiKeys = tuiKeys.filter((key) => typeof tuiEnglishValues[key] !== 'string')
const missingDesktopKeys = desktopKeys.filter((key) => {
  const value = desktopEnglishValues[key]
  return typeof value !== 'string' && !(Array.isArray(value) && value.every((item) => typeof item === 'string'))
})
const romanianTui = Object.fromEntries(tuiKeys.map((key) => {
  const value = tuiEnglishValues[key]
  if (typeof value !== 'string') {
    console.warn(`No English TUI source for ${key}; preserving its key as a fallback.`)
    return [key, key]
  }
  return [key, translations.get(value) ?? value]
}))
const romanianDesktop = Object.fromEntries(desktopKeys.map((key) => {
  const value = desktopEnglishValues[key]
  if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
    return [key, JSON.stringify(value.map((item) => translations.get(item) ?? item))]
  }
  if (typeof value !== 'string') {
    console.warn(`No English desktop source for ${key}; preserving its key as a fallback.`)
    return [key, key]
  }
  return [key, translations.get(value) ?? value]
}))
const romanianWeb = applyWebEdits(webEnglish, webEdits, translations)

if (missingTuiKeys.length) console.warn(`Missing TUI English values (${missingTuiKeys.length}): ${missingTuiKeys.slice(0, 10).join(', ')}`)
if (missingDesktopKeys.length) console.warn(`Missing desktop English values (${missingDesktopKeys.length}): ${missingDesktopKeys.slice(0, 10).join(', ')}`)
if (tuiBranching.length) console.warn(`TUI strings with argument-dependent branches (rendered with positional placeholders): ${tuiBranching.length}`)

const yamlOptions = { lineWidth: 0, indent: 2, defaultStringType: 'QUOTE_DOUBLE', defaultKeyType: 'PLAIN' }
await writeFile(path.join(localeDir, 'ro.yaml'), yaml.stringify(romanianYaml, yamlOptions), 'utf8')
await writeFile(path.join(localeDir, 'ro.tui.yaml'), yaml.stringify(romanianTui, yamlOptions), 'utf8')
await writeFile(path.join(localeDir, 'ro.desktop.yaml'), yaml.stringify(romanianDesktop, yamlOptions), 'utf8')
await writeFile(path.join(root, 'web/src/i18n/ro.ts'), romanianWeb, 'utf8')

console.log(`Generated ${Object.keys(englishYamlValues).length} core strings, ${tuiKeys.length} TUI keys, ${desktopKeys.length} desktop keys, and ${webEdits.length} web strings.`)
