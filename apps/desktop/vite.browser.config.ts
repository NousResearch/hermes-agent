import { fileURLToPath } from 'node:url'

import type { Plugin } from 'vite'
import { defineConfig, loadConfigFromFile, mergeConfig } from 'vite'

const root = fileURLToPath(new URL('.', import.meta.url)).replaceAll('\\', '/')

export default defineConfig(async (environment) => {
  const desktop = await loadConfigFromFile(environment, `${root}vite.config.ts`, root)

  if (!desktop) {throw new Error('Desktop Vite configuration could not be loaded.')}

  const plugins: Plugin[] = [{
    name: 'hermes-browser-identity',
    enforce: 'pre',
    resolveId(source, importer) {
      if (source === './preview-pane' && importer === `${root}src/app/chat/right-rail/preview.tsx`) {
        return `${root}browser/preview-pane.tsx`
      }
    },
    generateBundle() {
      for (const id of this.getModuleIds()) {
        if (id.endsWith('/src/lib/mcp-dashboard-oauth.ts') || id.endsWith('/src/app/chat/right-rail/preview-pane.tsx')) {
          this.error(`Native-only renderer module included in browser bundle: ${id}`)
        }
      }

      this.emitFile({ type: 'asset', fileName: 'hermes-frontend.json', source: JSON.stringify({ frontend: 'desktop' }) })
    }
  }, {
    name: 'hermes-browser-tailwind-source',
    enforce: 'pre',
    transform(code, id) {
      if (id.split('?')[0] !== `${root}src/styles.css`) {return null}

      return { code: `@source "${root}src";\n${code}`, map: null }
    }
  }]

  // Prepend the source transform so Tailwind sees it before compiling the CSS.
  return mergeConfig(mergeConfig({ plugins }, desktop.config), {
    resolve: { alias: [{ find: '@/lib/mcp-dashboard-oauth', replacement: `${root}browser/mcp-oauth.ts` }] },
    root: `${root}browser`,
    base: './',
    publicDir: `${root}public`,
    build: { outDir: `${root}dist/browser`, emptyOutDir: true }
  })
})
