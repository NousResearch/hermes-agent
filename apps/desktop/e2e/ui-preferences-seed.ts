import * as fs from 'node:fs'
import * as path from 'node:path'

/** The classic (pre-redesign) shell: the machine-shared rollback choice `store/shell-variant.ts` reads. */
export const CLASSIC_SHELL = { 'hermes.desktop.agentShell.v1': 'false' } as const

/**
 * Write the machine's `ui-preferences.json` (the on-disk contract in `electron/ui-preferences.ts`)
 * before launch. Machine-shared keys hydrate from this file on the first paint, so a value set only in
 * localStorage would be overwritten by whatever the store holds.
 */
export function seedUiPreferences(hermesHome: string, values: Record<string, string>): void {
  fs.writeFileSync(
    path.join(hermesHome, 'ui-preferences.json'),
    JSON.stringify({ version: 1, generation: 'e2e-seed', revision: 1, values })
  )
}
