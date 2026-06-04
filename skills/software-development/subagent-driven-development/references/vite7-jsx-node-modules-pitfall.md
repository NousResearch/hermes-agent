# Vite 7 + JSX in node_modules — Build Failure

**Symptom:** `npm run build` fails with `[commonjs--resolver] Expression expected` pointing to a `.js` file inside `node_modules/`. The file contains raw JSX (e.g., `<MediaPlayerBridge {...props}`).

**Root cause:** Some npm packages ship JSX in their production bundles (`.js` extension). Vite 7 uses Rollup 4 which has a strict parser — it encounters JSX in a `.js` file and fails. The `@vitejs/plugin-react` doesn't help because Rollup's commonjs resolver parses the file BEFORE the react plugin's transform hook runs.

**Confirmed affected packages (as of June 2026):**
- `@vidstack/react@1.15.2` — `prod/vidstack.js` line 226 contains `<MediaPlayerBridge {...props}...`

**What does NOT work:**
- `react({ include: [/\.tsx?$/, /@vidstack.*\.js$/] })` — Rollup parses first
- `optimizeDeps.include: ['@vidstack/react']` — only affects dev server
- `build.commonjsOptions.transformMixedEsModules: true` — doesn't help with JSX
- `ssr.noExternal: ['@vidstack/react']` — SSR-only config, doesn't affect client build

**Workarounds:**

1. **Use a different library** (preferred). For vidstack specifically, build a custom video player with native HTML5 `<video>` + React controls. The existing `MediaPlayer.tsx` pattern works well.

2. **Pre-bundle the problematic package** with esbuild before Vite processes it. Add a custom plugin:
   ```ts
   // vite.config.ts — not tested, theoretical
   import esbuild from 'esbuild';
   // ... custom plugin that transforms .js files from the package through esbuild
   ```

3. **Downgrade to Vite 6** if the library is essential and no alternative exists.

4. **Patch the package** — post-install script to strip JSX via esbuild transform. Fragile.

**Detection:** If `npm run build` fails with `Expression expected` pointing to a node_modules `.js` file, check if it contains JSX:
```bash
grep -n "return <" node_modules/<package>/prod/<file>.js | head -5
```

**Rule:** Before adopting a new media/UI library in a Vite 7 project, check if its production bundle contains JSX:
```bash
npm install <package>
grep -rn "return <\|<[A-Z]" node_modules/<package>/prod/ 2>/dev/null | head -5
```
