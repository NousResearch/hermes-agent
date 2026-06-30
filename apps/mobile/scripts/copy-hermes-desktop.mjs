#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(here, '..');
const defaultHermesRoot = resolve(appRoot, '..', '..');
const hermesRoot = process.env.HERMES_AGENT_REPO ? resolve(process.env.HERMES_AGENT_REPO) : defaultHermesRoot;
const desktopSrc = resolve(hermesRoot, 'apps/desktop/src');

if (!existsSync(desktopSrc)) {
  throw new Error(`Hermes Desktop source not found at ${desktopSrc}. Run this inside the hermes-agent repo or set HERMES_AGENT_REPO.`);
}

const target = resolve(appRoot, 'src');
rmSync(target, { recursive: true, force: true });
mkdirSync(target, { recursive: true });
cpSync(desktopSrc, target, { recursive: true, dereference: true });

// Expo Router treats any src/app directory as route files. Hermes Desktop's
// renderer folder is named src/app, so move it intact and preserve imports via
// tsconfig alias @/app/* -> src/hermes-app/*.
const desktopAppDir = resolve(target, 'app');
const mobileSafeAppDir = resolve(target, 'hermes-app');
if (existsSync(desktopAppDir)) {
  rmSync(mobileSafeAppDir, { recursive: true, force: true });
  cpSync(desktopAppDir, mobileSafeAppDir, { recursive: true, dereference: true });
  rmSync(desktopAppDir, { recursive: true, force: true });
}

console.log(`Copied Hermes Desktop renderer source from ${desktopSrc} -> ${target}`);
console.log('Moved copied Desktop src/app -> src/hermes-app to avoid Expo Router route detection.');
