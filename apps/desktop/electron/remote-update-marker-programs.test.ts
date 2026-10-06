import assert from 'node:assert/strict'
import { execFile as execFileCallback } from 'node:child_process'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { REMOTE_MARKER_JUDGE_PY } from './remote-update-marker-programs'

const execFile = promisify(execFileCallback)
const corpusPath = path.resolve(__dirname, '../../../tests/fixtures/update_marker_corpus.json')

// Feed every corpus judge case, with its injected process table, through the
// exact Python an SSH remote runs: the remote reader obeys the shared contract.
const DRIVER = String.raw`
import json
corpus=json.load(open(sys.argv[1],encoding='utf-8'))
out={}
for case in corpus['judge']:
    live={int(pid):ct for pid,ct in case['live'].items()}
    own=case.get('our_ct',corpus['our_ct'])
    env={'our_pid':case.get('our_pid',corpus['our_pid']),'our_ct':lambda own=own:own,'alive':lambda pid,live=live:pid in live,'ct':lambda pid,live=live:live.get(pid),'now':corpus['now']}
    verdict,owner=marker_judge(case['text'],env)
    out[case['name']]={'verdict':verdict,'owner':owner}
print(json.dumps(out))
`

test.skipIf(process.platform === 'win32')('the remote marker judge agrees with every corpus judge case', async () => {
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'))
  const { stdout } = await execFile('python3', ['-c', `${REMOTE_MARKER_JUDGE_PY}\n${DRIVER}`, corpusPath])

  const expected = Object.fromEntries(
    corpus.judge.map((c: any) => [c.name, { verdict: c.expect.verdict, owner: c.expect.owner }])
  )

  assert.ok(corpus.judge.length >= 40)
  assert.deepEqual(JSON.parse(stdout), expected)
})
