import { createRequire } from 'node:module'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import babel from '@rolldown/plugin-babel'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

const here = dirname(fileURLToPath(import.meta.url))
const upstreamRoot = resolve(here, '../../..')
const desktopRoot = resolve(upstreamRoot, 'apps/desktop')
const desktopSrc = resolve(desktopRoot, 'src')
const sharedSrc = resolve(upstreamRoot, 'apps/shared/src')
const requireFromDesktop = createRequire(resolve(desktopRoot, 'package.json'))
const reactDir = dirname(requireFromDesktop.resolve('react/package.json'))
const reactDomDir = dirname(requireFromDesktop.resolve('react-dom/package.json'))
const driverIife = resolve(dirname(requireFromDesktop.resolve('driver.js')), 'driver.js.iife.js')
const uiFonts = resolve(dirname(dirname(requireFromDesktop.resolve('@nous-research/ui/styles/fonts.css'))), 'fonts')

const backendUrl = process.env.HERMES_WEB_BACKEND_URL || 'http://127.0.0.1:13041'

function webTailwindSource() {
  return {
    name: 'hermes-web-tailwind-source',
    enforce: 'pre' as const,
    transform(code: string, id: string) {
      if (id.split('?')[0] !== resolve(desktopSrc, 'styles.css')) return null
      return {
        code: `@source "${desktopSrc.replaceAll('\\', '/')}";\n${code}`,
        map: null
      }
    }
  }
}

function upstreamFonts() {
  return {
    name: 'hermes-upstream-fonts',
    configureServer(server: { middlewares: { use: (route: string, handler: (request: any, response: any, next: () => void) => void) => void } }) {
      server.middlewares.use('/fonts', (request, response, next) => {
        const name = decodeURIComponent((request.url || '').replace(/^\//, '').split('?')[0])
        const path = resolve(uiFonts, name)
        if (!name || name.includes('/') || name.includes('\\') || !name.endsWith('.woff2') || !path.startsWith(`${uiFonts}/`) || !existsSync(path) || !statSync(path).isFile()) {
          next()
          return
        }
        response.setHeader('content-type', 'font/woff2')
        createReadStream(path).pipe(response)
      })
    }
  }
}

function compilerPreset() {
  const preset = reactCompilerPreset()
  if (!preset.rolldown.filter) {
    throw new Error('React compiler preset did not provide a Rolldown filter')
  }
  preset.rolldown.filter.code = /\/>|<\/|from\s*['"][^'"]*react/
  return preset
}

export default defineConfig(({ command }) => ({
  base: '/',
  publicDir: resolve(desktopRoot, 'public'),
  plugins: [webTailwindSource(), react(), babel({ presets: [compilerPreset()] }), tailwindcss(), upstreamFonts()],
  css: { postcss: { plugins: [] } },
  optimizeDeps: {
    exclude: [
      'driver.js',
      'driver.js/dist/driver.js.iife.js',
      'driver.js/dist/driver.js.iife.js?raw',
      'driver.js/dist/driver.css?raw'
    ]
  },
  resolve: {
    alias: {
      '@/debug/dev-only': command === 'serve'
        ? resolve(desktopSrc, 'debug/dev-only.ts')
        : resolve(desktopSrc, 'debug/dev-only.noop.ts'),
      '@desktop': desktopSrc,
      '@': desktopSrc,
      '@hermes/plugin-sdk': resolve(desktopSrc, 'sdk/index.ts'),
      '@hermes/shared/billing': resolve(sharedSrc, 'billing-types.ts'),
      '@hermes/shared/translucency': resolve(sharedSrc, 'translucency.ts'),
      '@hermes/shared': sharedSrc,
      'driver.js/dist/driver.js.iife.js?raw': `${driverIife}?raw`,
      'driver.js/dist/driver.js.iife.js': driverIife,
      react: reactDir,
      'react-dom': reactDomDir,
      'react/jsx-dev-runtime': resolve(reactDir, 'jsx-dev-runtime.js'),
      'react/jsx-runtime': resolve(reactDir, 'jsx-runtime.js')
    },
    dedupe: ['react', 'react-dom', 'react-router']
  },
  server: {
    host: '127.0.0.1',
    port: 5175,
    strictPort: true,
    fs: { allow: [here, upstreamRoot] },
    proxy: {
      '/api': { target: backendUrl, changeOrigin: true, ws: true },
      '/auth': { target: backendUrl, changeOrigin: true },
      '/login': { target: backendUrl, changeOrigin: true },
      '/logout': { target: backendUrl, changeOrigin: true }
    }
  },
  build: {
    outDir: resolve(here, 'dist'),
    emptyOutDir: true,
    chunkSizeWarningLimit: 25000
  }
}))
