import assert from 'node:assert/strict'

import { test } from 'vitest'

import { classifyExternalUrl, classifyResponseLinkUrl, wslExternalOpenCommand } from './external-url-policy'

test('classifyExternalUrl preserves trusted arbitrary local artifact URLs', () => {
  const result = classifyExternalUrl('file:///C:/Users/example/script.py')

  assert.equal(result?.kind, 'file')
  assert.equal(result?.url.protocol, 'file:')
})

test('classifyResponseLinkUrl accepts passive local files and Obsidian open-note URLs', () => {
  const fileResult = classifyResponseLinkUrl('file:///C:/Users/example/My%20Note.md')
  const obsidianResult = classifyResponseLinkUrl('obsidian://open?vault=Personal&file=00%20Inbox%2FMy%20Note.md')

  assert.equal(fileResult?.kind, 'file')
  assert.equal(fileResult?.url.protocol, 'file:')
  assert.equal(obsidianResult?.kind, 'external')
  assert.equal(obsidianResult?.url.protocol, 'obsidian:')
  assert.equal(obsidianResult?.url.hostname, 'open')
  assert.equal(classifyExternalUrl('obsidian://open?vault=Personal&file=Note.md'), null)
})

test('classifyExternalUrl rejects other application protocols and Obsidian actions', () => {
  assert.equal(classifyExternalUrl('file://attacker/share/note.md'), null)
  assert.equal(classifyExternalUrl('file:////attacker/share/note.md'), null)
  assert.equal(classifyExternalUrl('file://///attacker/share/note.md'), null)
  assert.equal(classifyExternalUrl('file://localhost//attacker/share/note.md'), null)
  assert.equal(classifyExternalUrl('file:///C:/Users/example/note%2Fescape.md'), null)
  assert.equal(classifyExternalUrl('file:///C:/Users/example/note%5Cescape.md'), null)
  assert.equal(classifyResponseLinkUrl('file:///C:/Users/example/note%2Fescape.md'), null)
  assert.equal(classifyResponseLinkUrl('file:///C:/Users/example/note%5Cescape.md'), null)
  assert.equal(classifyExternalUrl('vscode://file/C:/Users/example/note.md'), null)
  assert.equal(classifyExternalUrl('javascript:alert(1)'), null)
  assert.equal(classifyExternalUrl('obsidian://new?vault=Personal&name=Injected'), null)
  assert.equal(classifyExternalUrl('obsidian://open@attacker?vault=Personal'), null)
  assert.equal(classifyExternalUrl('obsidian://open/other-action?vault=Personal'), null)
})

test.each([
  'file:///C:/Users/example/payload.exe',
  'file:///C:/Users/example/payload.cmd',
  'file:///C:/Users/example/payload.bat',
  'file:///C:/Users/example/installer.msi',
  'file:///C:/Users/example/shortcut.lnk',
  'file:///C:/Users/example/website.url',
  'file:///C:/Users/example/script.ps1',
  'file:///C:/Users/example/document.pdf.exe',
  'file:///C:/Users/example/encoded%2Eexe'
])('classifyResponseLinkUrl rejects active response file targets: %s', target => {
  assert.equal(classifyResponseLinkUrl(target), null)
})

test('wslExternalOpenCommand bypasses cmd parsing and preserves URL metacharacters', () => {
  const url = 'obsidian://open?vault=Personal&file=00%20Inbox%2FNote%20(1)%5E%7C%3C%3E.md'
  const command = wslExternalOpenCommand(url)

  assert.equal(command.executable, 'rundll32.exe')
  assert.deepEqual(command.args, ['url.dll,FileProtocolHandler', url])
  assert.equal(command.args.length, 2)
  assert.ok(!command.args.some(argument => /cmd\.exe|\/c|start/i.test(argument)))
})
