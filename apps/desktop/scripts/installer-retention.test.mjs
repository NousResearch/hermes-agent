import assert from 'node:assert/strict'
import fs from 'node:fs'

import { describe, test } from 'vitest'

const script = fs.readFileSync(new URL('../electron/installer-retention.nsh', import.meta.url), 'utf8')

function position(fragment) {
  const offset = script.indexOf(fragment)
  assert.notEqual(offset, -1, `missing NSIS retention guard: ${fragment}`)

  return offset
}

describe('retained rollback installer NSIS hook', () => {
  test('preserves the existing installer until the replacement copy is known-good', () => {
    const copyNew = position('CopyFiles /SILENT "$EXEPATH" "$retainedInstallerTemp"')
    const guardNewCopy = position('IfErrors retain_installer_cleanup_temp')
    const copyBackup = position('CopyFiles /SILENT "$retainedInstaller" "$retainedInstallerBackup"')
    const deleteExisting = position('Delete "$retainedInstaller"')
    const promoteNew = position('Rename "$retainedInstallerTemp" "$retainedInstaller"')
    const restoreBackup = position('CopyFiles /SILENT "$retainedInstallerBackup" "$retainedInstaller"')

    assert.ok(copyNew < guardNewCopy)
    assert.ok(guardNewCopy < copyBackup)
    assert.ok(copyBackup < deleteExisting)
    assert.ok(deleteExisting < promoteNew)
    assert.ok(promoteNew < restoreBackup)
  })

  test('checks replacement promotion and retains a backup when restoration fails', () => {
    position('IfErrors retain_installer_restore_backup')
    position('IfErrors retain_installer_done')
    position('Delete "$retainedInstallerBackup"')
  })
})
