import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'
vi.mock('electron', () => ({ app: {}, ipcMain: {} }))
import { preparedJournal } from './prepared-submissions'

test('acknowledged journal mutations survive reopening with exact payload and origin isolation', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-journal-'))

  try {
    const origin = 'http://localhost:5174'
    const key = JSON.stringify(['remote', 'profile-a', 'stored-a'])
    const entry = { id: 'id-a', text: 'Ω 👩🏽‍💻 e\u0301\n  []', attachments: [{ id: 'asset', path: '/tmp/Ω' }], params: { session_id: 'live-a' } }
    preparedJournal(dir, origin).update(key, entry)
    preparedJournal(dir, origin).update('other-profile', { id: 'id-b' })
    expect(preparedJournal(dir, origin).read()[key]).toEqual(entry)
    expect(preparedJournal(dir, 'http://other:5174').read()).toEqual({})
    preparedJournal(dir, origin).update(key, null)
    expect(preparedJournal(dir, origin).read()).toEqual({ 'other-profile': { id: 'id-b' } })
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
