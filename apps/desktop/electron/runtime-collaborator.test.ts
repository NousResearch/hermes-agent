import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { discoverCollaborators, readCollaboratorChoice, saveCollaboratorChoice } from './runtime-collaborator'

const roots: string[] = []
afterEach(() => roots.splice(0).forEach(root => fs.rmSync(root, { recursive: true, force: true })))

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'collaborator-test-'))
  roots.push(root)

  return root
}

test('persisted choice survives restarts, while unavailable or damaged choices return to setup', () => {
  const root = fixture()
  const file = path.join(root, 'profile', 'runtime-collaborator.json')
  expect(readCollaboratorChoice(file)).toBeNull()
  saveCollaboratorChoice(file, { mode: 'bundled' })
  expect(readCollaboratorChoice(file)).toEqual({ mode: 'bundled' })
  fs.mkdirSync(path.join(root, 'hermes_cli'))
  fs.writeFileSync(path.join(root, 'hermes_cli/main.py'), '')
  const python = path.join(root, 'python')
  fs.writeFileSync(python, '')
  const choice = { mode: 'existing' as const, root, python }
  saveCollaboratorChoice(file, choice)
  expect(readCollaboratorChoice(file)).toEqual(choice)
  fs.unlinkSync(python)
  expect(readCollaboratorChoice(file)).toBeNull()
  fs.writeFileSync(file, '{broken')
  expect(readCollaboratorChoice(file)).toBeNull()
  fs.writeFileSync(file, 'null')
  expect(readCollaboratorChoice(file)).toBeNull()
})

test('discovery does not offer an incomplete installation or install anything into it', async () => {
  const root = fixture()
  fs.mkdirSync(path.join(root, 'hermes_cli'))
  fs.writeFileSync(path.join(root, 'hermes_cli/main.py'), '')
  expect(await discoverCollaborators([root, root])).toEqual([])
  expect(fs.readdirSync(root)).toEqual(['hermes_cli'])
})
