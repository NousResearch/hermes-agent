const fs = require('fs')
const cs = fs.readFileSync('/tmp/sync/apps/desktop/src/i18n/cs.ts', 'utf8').split('\n')
const en = fs.readFileSync('/tmp/sync/apps/desktop/src/i18n/en.ts', 'utf8').split('\n')

const csAnchors = [
  "close: 'Zavřít'",
  'composerPopoutDesc',
  "toggleFailed:",
  'serverStates: {',
  'soulDesc:',
  'promptLabel:',
  'handoffOrigin:',
  'queueStuckBody',
  'todos: (done, total)',
  'readAloud:',
  'clarify: {'
]

function show(lines, re, before, after) {
  for (let i = 0; i < lines.length; i++) {
    if (re.test(lines[i])) {
      const s = Math.max(0, i - before), e = Math.min(lines.length, i + after)
      for (let j = s; j < e; j++) console.log(String(j + 1).padStart(5) + '| ' + lines[j])
      return true
    }
  }
  console.log('NOT FOUND ' + re)
  return false
}

for (const a of csAnchors) {
  console.log('=========== CS anchor: ' + a)
  show(cs, new RegExp(a.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), 3, 12)
  console.log()
}
