import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  css: { postcss: { plugins: [] } },
  build: {
    minify: false,
    sourcemap: 'inline',
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
  },
  resolve: {
    alias: [
      { find: '@/app', replacement: path.resolve(__dirname, 'src/hermes-app') },
      { find: 'react-arborist', replacement: path.resolve(__dirname, 'src/mobile-shims/react-arborist.tsx') },
      { find: 'react-shiki', replacement: path.resolve(__dirname, 'src/mobile-shims/react-shiki.tsx') },
      { find: '@streamdown/code', replacement: path.resolve(__dirname, 'src/mobile-shims/streamdown-code.ts') },
      { find: 'use-stick-to-bottom', replacement: path.resolve(__dirname, 'src/mobile-shims/use-stick-to-bottom.tsx') },
      { find: '@hermes/shared', replacement: path.resolve(__dirname, 'src/mobile-shims/hermes-shared') },
      { find: '@', replacement: path.resolve(__dirname, 'src') }
    ],
    dedupe: ['react', 'react-dom']
  },
  server: { host: '0.0.0.0', port: 5174, strictPort: true }
});
