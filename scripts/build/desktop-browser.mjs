#!/usr/bin/env node
// Reuse the Desktop dependency graph and Vite configuration without building Electron.
import { fileURLToPath, pathToFileURL } from 'node:url'
import { workspaceTool } from './frontend-common.mjs'

const root = fileURLToPath(new URL('../../', import.meta.url))
const { build } = await import(pathToFileURL(workspaceTool(root, 'apps/desktop', 'vite')).href)
await build({ configFile: `${root}apps/desktop/vite.browser.config.ts` })
