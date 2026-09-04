import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { DISMISSED_FILE, pendingNotice, recordDismissed, REPORT_FILE, reportKey } from './plugin-compat-notice'

function tmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-compat-'))
}

const REPORT = {
  removal_date: '2026-09-14',
  in_effect: false,
  plugins: {
    alpha: [
      {
        file: '__init__.py',
        line: 3,
        old: 'tools.web_tools.prefers_gateway',
        new: 'tools.tool_backend_helpers.prefers_gateway'
      }
    ],
    beta: [
      { file: 'a.py', line: 1, old: 'hermes_cli.kanban_db.connect', new: 'hermes_cli.kanban_db_connect.connect' },
      {
        file: 'a.py',
        line: 9,
        old: 'hermes_cli.kanban_db.connect_closing',
        new: 'hermes_cli.kanban_db_connect.connect_closing'
      }
    ]
  },
  lines: [
    '2 plugins use import paths that stop working on 2026-09-14 (10 days): alpha (1), beta (2)',
    'Details: hermes plugins compat'
  ]
}

test('no report file → no notice', () => {
  assert.equal(pendingNotice(tmp(), tmp()), null)
})

test('report → one notice naming plugins, date and the CLI command', () => {
  const home = tmp()
  fs.writeFileSync(path.join(home, REPORT_FILE), JSON.stringify(REPORT))
  const n = pendingNotice(home, tmp())
  assert.ok(n)
  assert.equal(n.title, 'Plugins need an update')
  assert.match(n.message, /2 plugins import module paths that stop working on 2026-09-14/)
  assert.match(
    n.detail,
    /• alpha — 1 import \(e\.g\. tools\.web_tools\.prefers_gateway → tools\.tool_backend_helpers\.prefers_gateway\)/
  )
  assert.match(n.detail, /• beta — 2 imports/)
  assert.match(n.detail, /hermes plugins compat/)
})

test('dismissal is remembered for the same report and forgotten for a different one', () => {
  const home = tmp()
  const userData = tmp()
  fs.writeFileSync(path.join(home, REPORT_FILE), JSON.stringify(REPORT))
  const first = pendingNotice(home, userData)
  assert.ok(first)
  recordDismissed(userData, first.key)
  assert.ok(fs.existsSync(path.join(userData, DISMISSED_FILE)))
  assert.equal(pendingNotice(home, userData), null, 'same report must not show twice')

  // a third affected plugin is new information
  const grown = {
    ...REPORT,
    plugins: { ...REPORT.plugins, gamma: [{ file: 'g.py', line: 1, old: 'x.y', new: 'z.y' }] }
  }

  fs.writeFileSync(path.join(home, REPORT_FILE), JSON.stringify(grown))
  const second = pendingNotice(home, userData)
  assert.ok(second)
  assert.notEqual(second.key, first.key)

  // the date passing (plugins now disabled) is new information too, with different wording
  const disabled = { ...REPORT, in_effect: true }
  fs.writeFileSync(path.join(home, REPORT_FILE), JSON.stringify(disabled))
  const third = pendingNotice(home, userData)
  assert.ok(third)
  assert.equal(third.title, 'Some plugins were not loaded')
  assert.match(third.detail, /allow_deprecated_imports/)
  assert.notEqual(reportKey(disabled as any), reportKey(REPORT as any))
})

test('empty or malformed report is ignored', () => {
  const home = tmp()
  fs.writeFileSync(path.join(home, REPORT_FILE), JSON.stringify({ ...REPORT, plugins: {} }))
  assert.equal(pendingNotice(home, tmp()), null)
  fs.writeFileSync(path.join(home, REPORT_FILE), '{not json')
  assert.equal(pendingNotice(home, tmp()), null)
})
