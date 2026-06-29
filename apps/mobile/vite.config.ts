import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import fs from 'node:fs';
import path from 'node:path';

const mobileSrc = path.resolve(__dirname, 'src');
const desktopSrc = path.resolve(__dirname, '../desktop/src');
const sharedSrc = path.resolve(__dirname, '../shared/src');

const RESOLVE_EXTS = ['', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.css', '/index.ts', '/index.tsx', '/index.js'];

function tryResolve(root: string, rel: string): string | null {
  for (const ext of RESOLVE_EXTS) {
    const candidate = path.join(root, rel) + ext;
    try {
      if (fs.statSync(candidate).isFile()) return candidate;
    } catch {}
  }
  return null;
}

function tryResolveAbs(absPath: string): string | null {
  for (const ext of RESOLVE_EXTS) {
    const candidate = absPath + ext;
    try {
      if (fs.statSync(candidate).isFile()) return candidate;
    } catch {}
  }
  return null;
}

// Transitional resolver: prefer mobile fork files, fall back to desktop.
// Handles both `@/...` alias paths AND relative imports from inside mobile/src/
// (so a deleted fork file resolves through to the desktop sibling).
// End state once the fork is empty: drop this and use plain `@/* → ../desktop/src/*`.
function mobileFallback(): Plugin {
  const mobileHermesAppPrefix = path.join(mobileSrc, 'hermes-app') + path.sep;
  const mobileSrcPrefix = mobileSrc + path.sep;

  return {
    name: 'hermes-mobile-fallback',
    enforce: 'pre',
    resolveId(source, importer) {
      if (source.startsWith('@/')) {
        const rel = source.slice(2);
        if (rel === 'app' || rel.startsWith('app/')) {
          const sub = rel === 'app' ? '' : rel.slice(4);
          return (
            tryResolve(mobileSrc, path.join('hermes-app', sub)) ??
            tryResolve(desktopSrc, path.join('app', sub))
          );
        }
        return tryResolve(mobileSrc, rel) ?? tryResolve(desktopSrc, rel);
      }

      let absCandidate: string | null = null;
      if (path.isAbsolute(source)) {
        absCandidate = source;
      } else if (importer && (source.startsWith('./') || source.startsWith('../'))) {
        absCandidate = path.resolve(path.dirname(importer), source);
      }
      if (absCandidate) {
        if (tryResolveAbs(absCandidate)) return null;
        if (absCandidate.startsWith(mobileHermesAppPrefix)) {
          return tryResolve(desktopSrc, path.join('app', absCandidate.slice(mobileHermesAppPrefix.length)));
        }
        if (absCandidate.startsWith(mobileSrcPrefix)) {
          return tryResolve(desktopSrc, absCandidate.slice(mobileSrcPrefix.length));
        }
      }

      return null;
    }
  };
}

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), mobileFallback()],
  css: { postcss: { plugins: [] } },
  build: {
    minify: false,
    sourcemap: 'inline',
    rollupOptions: { output: { inlineDynamicImports: true } }
  },
  resolve: {
    alias: [
      // Mobile-only shims for libraries with desktop-incompatible runtime needs.
      { find: 'react-arborist', replacement: path.resolve(mobileSrc, 'mobile-shims/react-arborist.tsx') },
      { find: 'react-shiki', replacement: path.resolve(mobileSrc, 'mobile-shims/react-shiki.tsx') },
      { find: 'use-stick-to-bottom', replacement: path.resolve(mobileSrc, 'mobile-shims/use-stick-to-bottom.tsx') },
      // Real shared package — no fork.
      { find: '@hermes/shared', replacement: sharedSrc }
    ],
    dedupe: ['react', 'react-dom']
  },
  server: { host: '0.0.0.0', port: 5174, strictPort: true }
});
