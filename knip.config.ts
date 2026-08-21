import type { KnipConfig } from 'knip'

/**
 * Knip finds unused files, exports and dependencies across the npm workspaces.
 *
 * Three things need spelling out that Knip cannot infer on its own:
 *
 *  1. Tailwind v4 keeps a large share of our dependency surface in CSS
 *     (`@import 'tailwindcss'`, `@plugin '@tailwindcss/typography'`, …). Knip
 *     treats `.css` as a foreign file type, so a compiler turns those at-rules
 *     into import statements it can resolve, and the workspaces that ship
 *     stylesheets add `**\/*.css` to their project patterns.
 *  2. Several entry points are invoked by something other than a bundler —
 *     esbuild (`electron/main.ts`), electron-builder hooks, Playwright, and a
 *     pile of hand-run `node scripts/*.mjs` diagnostics. They are listed per
 *     workspace; the diagnostics are kept deliberately, so they count as entry
 *     points rather than dead files.
 *  3. A handful of imports point at OS binaries or at internals of packages we
 *     already depend on. Those are ignored explicitly.
 *
 * The root workspace covers only the shared ESLint config. `plugins/` holds
 * built dashboard bundles and sidecars the Python agent spawns, and `website/`
 * is a standalone Docusaurus project with its own lockfile and its own install,
 * so neither has a resolvable dependency graph from here.
 */

const SOURCE_FILES = '**/*.{js,mjs,cjs,jsx,ts,tsx,mts,cts}'

const CSS_REFERENCE =
  /@(?:import|use|plugin|reference|source)\s+['"](.+?)['"]|\burl\(\s*['"]?(.+?)['"]?\s*\)/g

const cssImports = (text: string) =>
  [
    ...text
      // Our stylesheets document their own `@import`/`@source`/`url()` paths in
      // block comments; matching those produces phantom unresolved imports.
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .matchAll(CSS_REFERENCE)
  ]
    .map(match => match[1] ?? match[2])
    // `@font-face { src: url('../../../node_modules/@scope/pkg/…') }` is a real
    // use of that package; rewrite it back to the bare specifier so Knip
    // attributes it instead of chasing a relative path out of the workspace.
    .map(specifier => specifier.replace(/^(?:\.\.\/)*node_modules\//, ''))
    .filter(specifier => !specifier.startsWith('data:'))
    .map(specifier => `import '${specifier}'`)
    .join('\n')

const config: KnipConfig = {
  compilers: {
    css: cssImports
  },
  // Two vendored trees keep their upstream export surface on purpose — the
  // shadcn/ui primitives so `npx shadcn add` stays a clean diff, and the ink
  // fork so the next rebase stays reviewable. Dead *files* and dependencies in
  // them are still reported; only the export-level checks are muted.
  ignoreIssues: {
    'apps/desktop/src/components/ui/**': ['duplicates', 'exports', 'types'],
    'ui-tui/packages/hermes-ink/**': ['duplicates', 'exports', 'types'],
    // Imported only as `import('../lib/forceTruecolor.js?t=' + n)` — the tests
    // re-import it per case to get a fresh module. Knip cannot follow the
    // cache-busting query, so it sees the exports as unreachable.
    'ui-tui/src/lib/forceTruecolor.ts': ['exports']
  },
  workspaces: {
    '.': {
      entry: ['eslint.config.shared.mjs'],
      project: ['**/*.css'],
      ignore: ['plugins/**', 'website/**']
    },
    'apps/bootstrap-installer': {
      entry: ['index.html'],
      project: [SOURCE_FILES, '**/*.css'],
      // src/styles.css defers wholesale to apps/desktop/src/styles.css, so the
      // at-rules and @font-face urls that pull these in live in the desktop
      // workspace. They are still this app's own build inputs.
      ignoreDependencies: [
        '@nous-research/ui',
        '@tailwindcss/typography',
        '@vscode/codicons',
        'katex',
        'tailwindcss',
        'tw-shimmer'
      ]
    },
    'apps/desktop': {
      entry: [
        'index.html',
        'electron/main.ts',
        'electron/preload.ts',
        'electron/preview-reach.e2e.mts',
        'e2e/**/*.ts',
        'scripts/**/*.{mjs,ts}',
        'src/plugins/*/tests/*.test.mjs',
        // Swapped in for src/debug/dev-only.ts by the `@/debug/dev-only` alias
        // in vite.config.ts whenever the counters are compiled out.
        'src/debug/dev-only.noop.ts',
        // Copied into a scratch checkout by scripts/run-short-session-hang-repro.mjs.
        'src/app/chat/short-session-hang-repro.tsx'
      ],
      project: [SOURCE_FILES, '**/*.css'],
      ignoreDependencies: [
        // Deep imports into packages electron-builder already brings along, and
        // node-pre-gyp's CLI, invoked to rebuild native modules.
        '@electron/asar',
        '@mapbox/node-pre-gyp',
        'app-builder-lib',
        // Served straight off disk: vite.config.ts resolves the package
        // directory and mounts it, so no module ever imports the specifier.
        'emojibase-data'
      ],
      // Probed at runtime on the host OS, never installed by us.
      ignoreBinaries: ['cage', 'fc-cache', 'reg', 'sample', 'uv', 'wsl\\.exe']
    },
    'apps/shared': {
      entry: ['src/**/*.test.ts', 'src/**/*.test-d.ts']
    },
    'ui-tui': {
      entry: ['scripts/**/*.{mjs,ts,tsx}'],
      // scripts/visual borrows Electron from the desktop workspace on purpose,
      // rather than pulling a second ~100 MB copy into this one.
      ignoreDependencies: ['electron']
    },
    web: {
      entry: ['index.html'],
      project: [SOURCE_FILES, '**/*.css'],
      // Ambient declarations for `window.__HERMES_PLUGIN_SDK__`: pulled in by
      // tsc, never imported, and the type surface plugin authors code against.
      ignore: ['src/plugins/sdk.d.ts']
    },
    'tests-js': {
      entry: ['*.test.ts']
    }
  }
}

export default config
