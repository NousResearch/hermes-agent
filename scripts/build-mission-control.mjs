#!/usr/bin/env node
/**
 * Build Mission Control.
 *
 * Produces two bundles into `dist/`:
 *
 *   dist/index.js  — the plugin. React is EXTERNAL and resolved at runtime from
 *                    `window.__HERMES_PLUGIN_SDK__`, so the dashboard renders it
 *                    with its own single React instance. Bundling a second copy
 *                    would break every hook call in the tree.
 *   dist/host.js   — the standalone server's host. This one DOES bundle React,
 *                    because it is the host: it exposes the SDK globals and
 *                    mounts the registered root itself.
 *   dist/style.css — tokens + component styles, concatenated and minified.
 *
 * Run: node scripts/build-mission-control.mjs [--watch]
 */
import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const BUNDLE = path.join(ROOT, 'plugins', 'samagent', 'dashboard', 'ui_bundle');
const DIST = path.join(ROOT, 'plugins', 'samagent', 'dashboard', 'dist');

mkdirSync(DIST, { recursive: true });

const watch = process.argv.includes('--watch');
const dev = watch || process.argv.includes('--dev');

/* The SDK bridge module is the only thing that touches React from the plugin
   bundle. esbuild shims it to read the globals, so React stays external. */
const shim = {
  name: 'hermes-sdk-global',
  setup(build) {
    build.onResolve({ filter: /^react$/ }, () => ({ path: 'react', namespace: 'hermes-sdk-global' }));
    build.onResolve({ filter: /^react-dom\/client$/ }, () => ({ path: 'react-dom-client', namespace: 'hermes-sdk-global' }));
    build.onLoad({ filter: /.*/, namespace: 'hermes-sdk-global' }, (args) => {
      if (args.path === 'react') {
        return {
          contents: `
            const s = (typeof window !== 'undefined' && window.__HERMES_PLUGIN_SDK__) || {};
            export default s.React;
            export const createElement = s.React.createElement;
            export const Fragment = s.React.Fragment;
            export const useState = s.hooks.useState;
            export const useEffect = s.hooks.useEffect;
            export const useLayoutEffect = s.hooks.useLayoutEffect || s.hooks.useEffect;
            export const useCallback = s.hooks.useCallback;
            export const useMemo = s.hooks.useMemo;
            export const useRef = s.hooks.useRef;
            export const useContext = s.hooks.useContext;
            export const createContext = s.hooks.createContext;
            export const memo = s.React.memo;
          `,
          loader: 'js',
        };
      }
      return {
        contents: `
          const s = (typeof window !== 'undefined' && window.__HERMES_PLUGIN_SDK__) || {};
          export const createRoot = s.createRoot;
          export const hydrateRoot = s.hydrateRoot;
        `,
        loader: 'js',
      };
    });
  },
};

const common = {
  bundle: true,
  format: 'iife',
  target: ['chrome110', 'edge110', 'firefox115', 'safari16'],
  legalComments: 'none',
  logLevel: 'info',
  minify: !dev,
  sourcemap: dev ? 'inline' : false,
  define: { 'process.env.NODE_ENV': dev ? '"development"' : '"production"' },
};

async function buildCss() {
  const srcDir = path.join(BUNDLE, 'src');
  const order = ['tokens.css'];
  const files = readdirSync(srcDir)
    .filter((f) => f.endsWith('.css') && !order.includes(f))
    .sort();
  const parts = order
    .concat(files)
    .map((f) => `/* ---- ${f} ---- */\n${readFileSync(path.join(srcDir, f), 'utf8')}`);

  const out = parts.join('\n\n');
  if (!dev) {
    // Minimal, safe minification: strip comments and collapse whitespace.
    // No autoprefixer/transform games — the tokens use color-mix() and clamp()
    // deliberately, and both are baseline in the target browsers above.
    const min = out
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\s*([{}:;,>])\s*/g, '$1')
      .replace(/;}/g, '}')
      .replace(/\n\s*/g, '\n')
      .trim();
    writeFileSync(path.join(DIST, 'style.css'), min);
  } else {
    writeFileSync(path.join(DIST, 'style.css'), out);
  }
  return files.length + order.length;
}

async function main() {
  const n = await buildCss();

  const plugin = {
    ...common,
    entryPoints: [path.join(BUNDLE, 'src', 'index.js')],
    outfile: path.join(DIST, 'index.js'),
    plugins: [shim],
  };

  const host = {
    ...common,
    entryPoints: [path.join(BUNDLE, 'host.js')],
    outfile: path.join(DIST, 'host.js'),
    // The host IS React; it must not be shimmed.
  };

  if (watch) {
    const ctx1 = await esbuild.context(plugin);
    const ctx2 = await esbuild.context(host);
    await Promise.all([ctx1.watch(), ctx2.watch()]);
    console.log(`[mission-control] watching ${n} stylesheet(s) — dist/ updates on save`);
    return;
  }

  await esbuild.build(plugin);
  await esbuild.build(host);
  console.log(`[mission-control] built dist/index.js, dist/host.js, dist/style.css (${n} stylesheets)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
