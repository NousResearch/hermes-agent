// Paths and browser flags for the kit. Everything resolves from this folder, so the skill runs wherever it is installed.
// Override any of them with an environment variable.

import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));

export const FFMPEG = process.env.FFMPEG || 'ffmpeg';
export const FFPROBE = process.env.FFPROBE || 'ffprobe';
export const PYTHON = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

// three, esbuild, fonts and playwright: `npm install` in this folder puts them here.
export const DEPS = process.env.TEARDOWN_DEPS || path.join(ROOT, 'node_modules');
// Unset: playwright uses the Chromium that `npx playwright install chromium` downloaded.
export const CHROMIUM = process.env.CHROMIUM || undefined;
// Real GPU WebGL instead of a software rasteriser. On Windows d3d11 plus --force_high_performance_gpu keeps ANGLE off
// an integrated GPU when a discrete one exists. capture.mjs prints the GPU it got on every run.
export const CHROMIUM_ARGS = [
  ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : []),
  '--force_high_performance_gpu', '--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--enable-webgl', '--hide-scrollbars',
];

// Builds and renders land in ./teardown-out of the folder you run from.
export const OUT = path.resolve(process.env.TEARDOWN_OUT || 'teardown-out');
export const DIST = path.join(OUT, 'dist');

export const fromDeps = (id) => createRequire(path.join(DEPS, 'x.js'))(id);
export const playwright = () => fromDeps('playwright');
