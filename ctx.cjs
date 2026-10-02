const fs = require('fs')
const path = require('path')
const dir = '/tmp/sync/apps/desktop/src/i18n'
const en = fs.readFileSync(path.join(dir, 'en.ts'), 'utf8').split('\n')
const cs = fs.readFileSync(path.join(dir, 'cs.ts'), 'utf8').split('\n')

const anchors = [
  ['externalOpenFailed.missing', "missing: {", 'externalOpenFailed'],
  ['appearance.fileBrowser', "fileBrowserTitle", null],
  ['skills.toolset', "toolsetOn", null],
  ['skills.serverStates', "hermes_not_connected", null],
  ['profiles.soulMissing', "soulMissing", null],
  ['cron.script', "scriptLabel", null],
  ['sidebar.continuationOrigin', "continuationOrigin", null],
  ['composer.queueDropped', "queueDroppedTitle", null],
  ['statusStack.previousTodos', "previousTodos", null],
  ['assistant.thread.copyFull', "copyFullResponse", null],
  ['assistant.clarify.notDelivered', "notDelivered:", null]
]

function show(file, lines, re, before, after) {
  for (let i = 0; i < lines.length; i++) {
    if (re.test(lines[i])) {
      const s = Math.max(0, i - before), e = Math.min(lines.length, i + after)
      console.log('--- ' + file + ' @' + (i + 1))
      for (let j = s; j < e; j++) console.log(String(j + 1).padStart(5) + '| ' + lines[j])
      return
    }
  }
  console.log('--- ' + file + ' NOT FOUND for ' + re)
}

for (const [name, needle] of anchors) {
  console.log('====================== ' + name)
  show('EN', en, new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), 6, 12)
}
