/**
 * The File menu and renderer bridge live on opposite sides of Electron's
 * main-process boundary. Pin their source shapes so a future rename cannot
 * silently leave a visible menu item disconnected from its renderer action.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

const here = path.dirname(fileURLToPath(import.meta.url))

const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8')

const integrationsSource = fs.readFileSync(
  path.join(here, '..', 'src', 'app', 'contrib', 'hooks', 'use-desktop-integrations.ts'),
  'utf8'
)

test('File menu exposes New Session Tab through the main-process sender', () => {
  assert.match(mainSource, /\{ click: \(\) => sendNewTabRequested\(\), label: 'New Session Tab' \}/)
})

test('renderer subscribes to the New Session Tab bridge event', () => {
  assert.match(
    integrationsSource,
    /onNewSessionTabRequested\?\.\(\(\) => \$newSessionTabAction\.get\(\)\?\.\(\)\)/
  )
})
