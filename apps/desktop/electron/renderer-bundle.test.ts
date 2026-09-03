import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import {
  missingRendererAssets,
  parseLazyChunkRefs,
  parseModuleAssetRefs,
  type RendererBundleDeps
} from './renderer-bundle'

// A production-shaped index.html: the module entry Vite emits plus a
// modulepreload for a lazy chunk. These are the refs the browser fetches
// before any app code runs, and the ones a torn update leaves dangling.
const INDEX_HTML = [
  '<!doctype html>',
  '<html>',
  '  <head>',
  '    <script type="module" crossorigin src="/assets/index-a1b2c3.js"></script>',
  '    <link rel="modulepreload" href="/assets/shiki-block-COiz1pEN.js" />',
  '    <link rel="stylesheet" href="/assets/index-d4e5f6.css" />',
  '  </head>',
  '  <body><div id="root"></div></body>',
  '</html>'
].join('\n')

test('parseModuleAssetRefs collects module scripts and modulepreload hrefs', () => {
  assert.deepEqual(parseModuleAssetRefs(INDEX_HTML), ['/assets/index-a1b2c3.js', '/assets/shiki-block-COiz1pEN.js'])
})

test('parseModuleAssetRefs ignores non-module tags (plain stylesheet, non-module script)', () => {
  const html = [
    '<link rel="stylesheet" href="/assets/app.css" />',
    '<script src="/assets/analytics.js"></script>',
    '<script type="module" src="/assets/entry.js"></script>'
  ].join('\n')

  // Only the type="module" script is a boot-critical module ref; the stylesheet
  // and the classic (non-module) script are not part of the module graph.
  assert.deepEqual(parseModuleAssetRefs(html), ['/assets/entry.js'])
})

test('parseModuleAssetRefs drops absolute/CDN URLs — they are not this generation', () => {
  const html = [
    '<script type="module" src="https://cdn.example.com/vendor.js"></script>',
    '<link rel="modulepreload" href="//cdn.example.com/chunk.js" />',
    '<script type="module" src="/assets/local.js"></script>'
  ].join('\n')

  assert.deepEqual(parseModuleAssetRefs(html), ['/assets/local.js'])
})

test('parseModuleAssetRefs strips a leading ./ and any query/hash suffix', () => {
  const html = [
    '<script type="module" src="./assets/entry.js?v=123"></script>',
    '<link rel="modulepreload" href="./assets/lazy.js#frag" />'
  ].join('\n')

  assert.deepEqual(parseModuleAssetRefs(html), ['assets/entry.js', 'assets/lazy.js'])
})

test('parseModuleAssetRefs returns [] for empty/nullish/module-free html', () => {
  assert.deepEqual(parseModuleAssetRefs(''), [])
  assert.deepEqual(parseModuleAssetRefs(undefined as unknown as string), [])
  assert.deepEqual(parseModuleAssetRefs('<html><body>no modules here</body></html>'), [])
})

// Build a deps object whose fs is backed by an in-memory set of files that
// "exist beside index.html", so the intact/torn matrix is testable without a
// real bundle on disk.
function depsFor(indexDir: string, html: string, presentFiles: string[]): RendererBundleDeps {
  const present = new Set(presentFiles.map(f => path.join(indexDir, f)))

  return {
    readFileSync: () => html,
    existsSync: (file: string) => present.has(file)
  }
}

const INDEX_PATH = path.join('/app', 'dist', 'index.html')
const INDEX_DIR = path.dirname(INDEX_PATH)

test('missingRendererAssets: intact generation reports nothing missing', () => {
  const deps = depsFor(INDEX_DIR, INDEX_HTML, ['assets/index-a1b2c3.js', 'assets/shiki-block-COiz1pEN.js'])

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), [])
})

test('missingRendererAssets: torn generation names the dangling chunk', () => {
  // The exact real-world crash: index.html names shiki-block-COiz1pEN.js but the
  // update never wrote it beside index.html.
  const deps = depsFor(INDEX_DIR, INDEX_HTML, ['assets/index-a1b2c3.js'])

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), ['/assets/shiki-block-COiz1pEN.js'])
})

test('missingRendererAssets: a fully torn copy lists every referenced module', () => {
  const deps = depsFor(INDEX_DIR, INDEX_HTML, [])

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), [
    '/assets/index-a1b2c3.js',
    '/assets/shiki-block-COiz1pEN.js'
  ])
})

test('missingRendererAssets: existence is checked relative to the index dir, per copy', () => {
  // The same module name present next to one index but not the other is how a
  // split app.asar vs app.asar.unpacked package presents: one copy is intact,
  // the other is torn. Resolution must be dir-relative so the loader can prefer
  // the intact copy.
  const unpackedIndex = path.join('/app', 'app.asar.unpacked', 'dist', 'index.html')
  const unpackedDir = path.dirname(unpackedIndex)

  const intact = depsFor(unpackedDir, INDEX_HTML, ['assets/index-a1b2c3.js', 'assets/shiki-block-COiz1pEN.js'])

  const torn = depsFor(INDEX_DIR, INDEX_HTML, ['assets/index-a1b2c3.js'])

  assert.deepEqual(missingRendererAssets(unpackedIndex, intact), [])
  assert.deepEqual(missingRendererAssets(INDEX_PATH, torn), ['/assets/shiki-block-COiz1pEN.js'])
})

test('missingRendererAssets: an unreadable index is not treated as torn', () => {
  // A read failure is the existence gate's concern, not this check's — returning
  // [] here keeps the caller from skipping a copy it never actually inspected.
  const deps: RendererBundleDeps = {
    readFileSync: () => {
      throw new Error('EACCES: permission denied')
    },
    existsSync: () => false
  }

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), [])
})

test('missingRendererAssets: an index naming nothing checkable is not torn', () => {
  const deps = depsFor(INDEX_DIR, '<html><body>static shell, no modules</body></html>', [])

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), [])
})

// ---------------------------------------------------------------------------
// Lazy-chunk (__vite__mapDeps) awareness — #93479. index.html only names the
// boot-critical modules; the chunks behind React.lazy() routes (syntax-diff-*,
// shiki-*, mermaid-embed-*) live in each chunk's inline __vite__mapDeps table.
// A torn install whose preloads are intact but whose lazy chunks are missing
// used to pass the generation check and die on the first lazy import.
// ---------------------------------------------------------------------------

// Production-shaped Vite output: the deps table definition plus an index-only
// call site that must NOT be misread as a filename list.
const ENTRY_JS = [
  'const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=[',
  '"assets/syntax-diff-Bo0962zh.js","assets/shiki-block-COiz1pEN.js","assets/mermaid-embed-Cq7Xw2aa.js"',
  '])))=>i.map(i=>d[i]);',
  'const load=()=>import("./syntax-diff").then(m=>m.default);',
  '__vite__mapDeps([0,1])'
].join('\n')

test('parseLazyChunkRefs reads the __vite__mapDeps filename table, not call sites', () => {
  assert.deepEqual(parseLazyChunkRefs(ENTRY_JS), [
    'assets/syntax-diff-Bo0962zh.js',
    'assets/shiki-block-COiz1pEN.js',
    'assets/mermaid-embed-Cq7Xw2aa.js'
  ])
})

test('parseLazyChunkRefs returns [] for chunk-free/nullish js', () => {
  assert.deepEqual(parseLazyChunkRefs(''), [])
  assert.deepEqual(parseLazyChunkRefs(undefined as unknown as string), [])
  assert.deepEqual(parseLazyChunkRefs('console.log("no deps table here")'), [])
})

// deps helper for the graph walk: per-file contents, not one shared html.
function graphDepsFor(indexDir: string, files: Record<string, string>): RendererBundleDeps {
  const byPath = new Map(Object.entries(files).map(([rel, content]) => [path.join(indexDir, rel), content]))

  return {
    readFileSync: (file: string) => {
      const content = byPath.get(file)

      if (content === undefined) {
        throw new Error(`ENOENT: ${file}`)
      }

      return content
    },
    existsSync: (file: string) => byPath.has(file)
  }
}

const GRAPH_INDEX_HTML = [
  '<!doctype html>',
  '<script type="module" crossorigin src="./assets/index-a1b2c3.js"></script>'
].join('\n')

test('missingRendererAssets: generation with all lazy chunks present is intact', () => {
  const deps = graphDepsFor(INDEX_DIR, {
    'index.html': GRAPH_INDEX_HTML,
    'assets/index-a1b2c3.js': ENTRY_JS,
    'assets/syntax-diff-Bo0962zh.js': '// present',
    'assets/shiki-block-COiz1pEN.js': '// present',
    'assets/mermaid-embed-Cq7Xw2aa.js': '// present'
  })

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), [])
})

test('missingRendererAssets: torn lazy chunk is detected even when every preload exists', () => {
  // The exact #93479 shape: index.html's own module list checks out, but the
  // syntax-diff chunk behind React.lazy() never landed. Loading this copy
  // boots fine and then blanks the workspace on the first diff render.
  const deps = graphDepsFor(INDEX_DIR, {
    'index.html': GRAPH_INDEX_HTML,
    'assets/index-a1b2c3.js': ENTRY_JS,
    'assets/shiki-block-COiz1pEN.js': '// present',
    'assets/mermaid-embed-Cq7Xw2aa.js': '// present'
  })

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), ['assets/syntax-diff-Bo0962zh.js'])
})

test('missingRendererAssets: walks transitive map-deps without looping on cycles', () => {
  // A lazy chunk can carry its own deps table (shiki language chunks do).
  // The walk must follow it — and a mutual reference must not hang the boot.
  const deps = graphDepsFor(INDEX_DIR, {
    'index.html': GRAPH_INDEX_HTML,
    'assets/index-a1b2c3.js': [
      'const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["assets/level-one-aaa.js"])))=>i.map(i=>d[i]);'
    ].join('\n'),
    'assets/level-one-aaa.js': [
      'const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=[',
      '"assets/index-a1b2c3.js","assets/level-two-bbb.js"',
      '])))=>i.map(i=>d[i]);'
    ].join('\n')
  })

  assert.deepEqual(missingRendererAssets(INDEX_PATH, deps), ['assets/level-two-bbb.js'])
})

test('missingRendererAssets: a lazy-chunk-torn copy loses to an intact copy end to end', () => {
  // The resolver contract: the same generation check that orders app.asar vs
  // app.asar.unpacked must now see lazy-chunk tears too, so a torn candidate
  // is skipped instead of shipping a delayed "Failed to fetch dynamically
  // imported module" crash.
  const files = {
    'index.html': GRAPH_INDEX_HTML,
    'assets/index-a1b2c3.js': ENTRY_JS,
    'assets/shiki-block-COiz1pEN.js': '// present',
    'assets/mermaid-embed-Cq7Xw2aa.js': '// present'
  }

  const torn = graphDepsFor(INDEX_DIR, files)
  const intact = graphDepsFor(INDEX_DIR, { ...files, 'assets/syntax-diff-Bo0962zh.js': '// present' })

  assert.notDeepEqual(missingRendererAssets(INDEX_PATH, torn), [])
  assert.deepEqual(missingRendererAssets(INDEX_PATH, intact), [])
})
