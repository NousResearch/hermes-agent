import { defineConfig, searchForWorkspaceRoot } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'node:fs'
import path from 'node:path'

// Hermes Setup — Tauri-targeted Vite config.
//
// Port 5175 keeps us out of the way of:
//   web       (vite default 5173)
//   apps/desktop dev     (5174 per its package.json)
//
// `clearScreen: false` is the Tauri convention — they spawn vite as a child
// process and want our errors to stay visible.

const host = process.env.TAURI_DEV_HOST

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  clearScreen: false,
  server: {
    port: 5175,
    strictPort: true,
    host: host || '127.0.0.1',
    hmr: host
      ? {
          protocol: 'ws',
          host,
          port: 5176
        }
      : undefined,
    watch: {
      // Don't watch the Rust side — tauri-cli handles it.
      ignored: ['**/src-tauri/**']
    },
    fs: {
      // In a git worktree, node_modules is a symlink into the primary
      // checkout; Vite resolves the realpath and refuses to serve assets
      // (e.g. @nous-research/ui fonts) from outside the workspace root.
      // Allow the resolved node_modules alongside the normal root.
      allow: [
        searchForWorkspaceRoot(process.cwd()),
        fs.realpathSync(path.resolve(__dirname, '../../node_modules'))
      ]
    }
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
    emptyOutDir: true
  }
})
