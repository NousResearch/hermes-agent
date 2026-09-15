import assert from 'node:assert/strict'
import { test } from 'vitest'
import { appExecutionAliasExtensions } from './before-build.mjs'

test('one MSIX extension consumes the launchers declared by the payload', () => {
  const names = ['custom-cli', 'another-cli']
  const xml = appExecutionAliasExtensions(names)
  assert.equal((xml.match(/<uap5:Extension\b/g) ?? []).length, 1)
  for (const name of names) assert.ok(xml.includes(`Alias="${name}.exe"`))
  assert.ok(xml.includes('custom-cli.exe'))
  assert.ok(!xml.includes('windows.service'))
  assert.ok(!xml.includes('desktop6:Service'))
  assert.equal(appExecutionAliasExtensions([]), '')
})
