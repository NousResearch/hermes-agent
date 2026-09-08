import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

// #101543: New Group Chat must enable when the multi-source roster has ≥2
// selectable (non-ghost) bots. The create dialog already uses that set; the
// menu gate used to count only local-source rows via activeSourceRoster.

const rosterPane = readFileSync(join(process.cwd(), 'src/plugins/hermes-bots/roster-pane.tsx'), 'utf8')
const createDialog = readFileSync(join(process.cwd(), 'src/plugins/hermes-bots/create-dialog.tsx'), 'utf8')

describe('New Group Chat menu gate', () => {
  it('counts selectable bots across the full roster, matching the create dialog', () => {
    expect(rosterPane).toMatch(/disabled=\{roster\.filter\(bot => !bot\?\.ghost\)\.length < 2\}/)
    expect(rosterPane).not.toMatch(/disabled=\{activeSourceRoster\.length < 2\}/)
    expect(createDialog).toMatch(/selectableRoster = roster\.filter\(bot => !bot\?\.ghost\)/)
  })
})
