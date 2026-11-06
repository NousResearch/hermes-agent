import assert from 'node:assert/strict'
import { test } from 'vitest'
import { appExecutionAliasApplications, appExecutionAliasExtensions, xmlAttribute } from './before-build.mjs'

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

test('manifest fragments escape every interpolated attribute value', () => {
  const xml = appExecutionAliasApplications(['odd"&<name'], { appNamePascal: 'Hermes', displayName: 'Hermes & "Friends" <beta>' })
  const values = [...xml.matchAll(/="([^"]*)"/g)].map((m) => m[1].replace(/&#\d+;/g, ''))
  assert.ok(values.length > 0 && values.every((v) => !/[&<>"']/.test(v)), xml)
  assert.ok(xml.includes('DisplayName="Hermes &#38; &#34;Friends&#34; &#60;beta&#62;"'))
  assert.ok(xml.includes('Alias="odd&#34;&#38;&#60;name.exe"'))
  assert.equal(xmlAttribute('plain-name'), 'plain-name')
})
