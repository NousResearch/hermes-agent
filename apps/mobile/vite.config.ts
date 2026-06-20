import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Resolve a path relative to this config file.
const r = (p: string) => fileURLToPath(new URL(p, import.meta.url))

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      // The desktop renderer references itself via `@/...`; point that at the
      // desktop workspace source so its components run here UNMODIFIED — no
      // vendoring. The mobile bridge also uses `@/global` etc. via this alias.
      '@': r('../desktop/src'),
      // Shared gateway client (workspace package).
      '@hermes/shared': r('../shared/src/index.ts'),
      // Mobile-only (net-new) code.
      '~mobile': r('./src/mobile'),
      '~bridge': r('./src/bridge'),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5180,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
