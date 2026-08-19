import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import fs from 'fs'
import { createRequire } from 'module'

// `hgui` symlinks a worktree's node_modules to the main checkout. Vite realpaths
// those before enforcing server.fs.allow, so codicon/font assets resolve outside
// the worktree root and 404. Whitelist the real node_modules locations.
const real = (p: string): string | null => {
  try {
    return fs.realpathSync(p)
  } catch {
    return null
  }
}

const fsAllow = [
  ...new Set(
    [
      path.resolve(__dirname, '../..'),
      real(path.resolve(__dirname, 'node_modules')),
      real(path.resolve(__dirname, '../../node_modules'))
    ].filter((p): p is string => p !== null)
  )
]

// React refuses to run when `react` and `react-dom` come from two different
// installed copies ("Minified React error #527" — a blank window, since it
// throws before the first paint). Both packages are pinned to one version in
// this workspace's package.json, but npm hoists whatever *it* considers
// compatible to the monorepo root: a root dependency whose react peer is a
// loose range (e.g. `^18 || ^19`) pulls the newest react up there, while
// react-dom stays at the pinned one. `^19.2.7` accepts `19.2.8`, so npm never
// warns. Resolving from this workspace instead of a hardcoded root path yields
// the versions declared here — npm nests a copy under the workspace exactly
// when the hoisted one differs, so the pair can only ever match.
const requireFromApp = createRequire(path.join(__dirname, 'vite.config.ts'))
const reactDir = path.dirname(requireFromApp.resolve('react/package.json'))
const reactDomDir = path.dirname(requireFromApp.resolve('react-dom/package.json'))

// The dev-only render/state churn counters (src/debug) must be imported
// STATICALLY above react-dom — react-dom captures the devtools hook at module
// init, so a dynamic import lands too late and observes zero commits. A static
// side-effect import can't be tree-shaken, so instead the whole graph is
// aliased out of any non-dev build. `command === 'serve'` covers `vite dev`;
// the perf harness opts a production build back in with VITE_PERF_PROBE=1.
const debugEntry = (command: string, env: Record<string, string>) =>
  command === 'serve' || env.VITE_PERF_PROBE === '1'
    ? path.resolve(__dirname, './src/debug/dev-only.ts')
    : path.resolve(__dirname, './src/debug/dev-only.noop.ts')

// The emoji picker (frimousse) fetches `<emojibaseUrl>/<locale>/data.json` at
// runtime. Its default is a CDN; Electron must work offline, so serve the
// bundled emojibase-data package at a stable local path instead — middleware
// in dev, emitted assets in the build. Only the files a locale actually needs.
const emojibaseDir =
  real(path.resolve(__dirname, 'node_modules/emojibase-data')) ??
  real(path.resolve(__dirname, '../../node_modules/emojibase-data'))

const EMOJIBASE_PATH = /^[a-z-]+\/(data|messages|shortcodes\/emojibase)\.json$/

const emojibaseAssets = () => ({
  name: 'hermes:emojibase-assets',
  configureServer(server: {
    middlewares: { use: (route: string, handler: (req: any, res: any, next: () => void) => void) => void }
  }) {
    server.middlewares.use('/emojibase', (req, res, next) => {
      const rel = (req.url ?? '').split('?')[0].replace(/^\/+/, '')
      if (!emojibaseDir || !EMOJIBASE_PATH.test(rel)) return next()
      fs.readFile(path.join(emojibaseDir, rel), (err: unknown, buf: Buffer) => {
        if (err) return next()
        res.setHeader('Content-Type', 'application/json')
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        res.end(buf)
      })
    })
  },
  generateBundle(this: { emitFile: (asset: { type: 'asset'; fileName: string; source: Uint8Array }) => void }) {
    if (!emojibaseDir) return
    for (const rel of ['en/data.json', 'en/messages.json', 'en/shortcodes/emojibase.json']) {
      this.emitFile({
        type: 'asset',
        fileName: `emojibase/${rel}`,
        source: fs.readFileSync(path.join(emojibaseDir, rel))
      })
    }
  }
})

export default defineConfig(({ command }) => ({
  base: './',
  plugins: [react(), tailwindcss(), emojibaseAssets()],
  css: {
    // Pin an explicit (empty) PostCSS config. Tailwind is handled entirely by
    // `@tailwindcss/vite`, so the renderer needs no PostCSS plugins — and
    // without this, Vite's `postcss-load-config` walks UP the filesystem
    // looking for a stray `postcss.config.*` / `tailwind.config.*`. The desktop
    // build runs from inside the user's home tree (e.g.
    // `C:\Users\<name>\AppData\Local\hermes\hermes-agent\apps\desktop`), so an
    // unrelated Tailwind v3 config higher up the tree gets picked up and
    // reprocesses our v4 stylesheet, failing the build with
    // "`@layer base` is used but no matching `@tailwind base` directive is
    // present." Pinning the config makes the build hermetic.
    postcss: { plugins: [] }
  },
  build: {
    // The renderer intentionally ships FEW chunks (not one, not thousands):
    //   · `codeSplitting: false` (the old setup) inlines every `lazy()` /
    //     dynamic import into the entry, so heavyweight lazy-only deps
    //     (mermaid, shiki grammars, katex) are parsed + evaluated on every
    //     cold start even though nothing rendered them. By the time the
    //     bundle hit ~28 MB that eval was ~1s of launch on an M-series.
    //   · Default splitting emits a chunk per shiki grammar/theme — thousands
    //     of files, which electron-builder OOMs scanning (#38888).
    // `advancedChunks` is the middle ground: heavyweight libraries merge into
    // a handful of named vendor chunks loaded on first use, app-level dynamic
    // imports stay lazy, and the file count stays in the tens.
    chunkSizeWarningLimit: 25000,
    rolldownOptions: {
      output: {
        advancedChunks: {
          groups: [
            // Shared foundations FIRST (first match wins): an unmatched
            // module shared by the entry and a heavy chunk gets merged INTO
            // the heavy chunk, and the entry then statically imports 19 MB of
            // shiki just to reach react/hast utils — putting the heavy chunk
            // right back on the boot path.
            { name: 'vendor-react', test: /node_modules[\\/](react|react-dom|scheduler|react-router)[\\/]/ },
            {
              name: 'vendor-md',
              test: /node_modules[\\/](property-information|hast-util-[^\\/]+|mdast-util-[^\\/]+|micromark[^\\/]*|unist-util-[^\\/]+|vfile[^\\/]*|unified|stringify-entities|space-separated-tokens|comma-separated-tokens|zwitch|html-void-elements|devlop|style-to-js|style-to-object|clsx)[\\/]/
            },
            // Shared utility packages the entry ALSO uses — kept out of the
            // heavy groups for the same boot-path reason.
            {
              name: 'vendor-util',
              test: /node_modules[\\/](lodash-es|es-toolkit|uuid|dayjs|d3-array|d3-color|d3-force|d3-interpolate|d3-time[^\\/]*|dompurify|stylis)[\\/]/
            },
            // One chunk per heavyweight, lazy-only library family.
            // @streamdown/code lives WITH shiki because it statically imports
            // the full shiki bundle.
            {
              name: 'mermaid',
              test: /node_modules[\\/](mermaid|cytoscape|dagre|khroma|elkjs|d3|d3-[^\\/]+|@mermaid-js)[\\/]/
            },
            {
              name: 'shiki',
              test: /node_modules[\\/](shiki|@shikijs|react-shiki|@streamdown[\\/]code|oniguruma-to-es|oniguruma-parser|regex(-[^\\/]+)?)[\\/]/
            },
            { name: 'katex', test: /node_modules[\\/]katex[\\/]/ }
          ]
        }
      }
    }
  },
  // driver.js only enters the graph through the tour's DYNAMIC import chain
  // (lib/tour/run-tour.ts), so the dep scanner never sees it at startup. Left
  // alone, first use registers it as a missing dep at runtime — which (a)
  // esbuild-prebundles the `?raw` IIFE import as a JS module, breaking the
  // raw-text transform ("does not provide an export named 'default'"), and
  // (b) triggers Vite's "new dependencies optimized" full page reload mid-
  // session. It's pure ESM with no CJS deps, so serving it unoptimized is
  // free. Query and bare forms all listed — exclusion matches exact ids.
  optimizeDeps: {
    exclude: ['driver.js', 'driver.js/dist/driver.js.iife.js', 'driver.js/dist/driver.js.iife.js?raw', 'driver.js/dist/driver.css?raw']
  },
  resolve: {
    alias: {
      '@/debug/dev-only': debugEntry(command, process.env as Record<string, string>),
      '@': path.resolve(__dirname, './src'),
      '@hermes/plugin-sdk': path.resolve(__dirname, './src/sdk/index.ts'),
      '@hermes/shared/billing': path.resolve(__dirname, '../shared/src/billing-types.ts'),
      '@hermes/shared': path.resolve(__dirname, '../shared/src'),
      // The tour tool's preview surface injects driver.js's prebuilt IIFE into
      // the pane's guest page as raw source; the package's exports map doesn't
      // expose that dist file (nor ./package.json), so resolve the main entry
      // (dist/driver.js.cjs) and point at its sibling. Both keys on purpose:
      // alias matching is exact, and the id reaches it with the `?raw` query
      // still attached in dev but stripped in some build paths.
      'driver.js/dist/driver.js.iife.js?raw': `${path.join(
        path.dirname(requireFromApp.resolve('driver.js')),
        'driver.js.iife.js'
      )}?raw`,
      'driver.js/dist/driver.js.iife.js': path.join(
        path.dirname(requireFromApp.resolve('driver.js')),
        'driver.js.iife.js'
      ),
      react: reactDir,
      'react-dom': reactDomDir,
      'react/jsx-dev-runtime': path.join(reactDir, 'jsx-dev-runtime.js'),
      'react/jsx-runtime': path.join(reactDir, 'jsx-runtime.js')
    },
    dedupe: ['react', 'react-dom', 'react-router']
  },
  server: {
    host: '127.0.0.1',
    port: 5174,
    strictPort: true,
    fs: {
      allow: fsAllow
    }
  },
  preview: {
    host: '127.0.0.1',
    port: 4174
  }
}))
