import { getHermesConfigRecord, saveHermesConfigRecord } from '@/hermes'
import { translateNow } from '@/i18n'
import { registryConnectionKind } from '@/store/connection-registry-state'
import { dialedGatewayModeFor } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import type { OnboardingScope } from '@/store/onboarding-scope'
import type { HermesConfigRecord } from '@/types/hermes'

// What a non-technical first run gets when it picks Simple: file edits inside a
// folder can be rolled back, no repo-coding brief in a chat that is not about
// code, and a hard stop on common forms of a few commands that destroy work no
// undo can bring back. The deny globs are matched case-insensitively with
// fnmatch against each command segment starting at its executable
// (tools/approval_floors.py), so they are position-independent: flags after
// the refspec, `+ref` force-pushes and `git -C <dir>` all match, `--follow-tags`
// does not. Root/home `rm -rf`, mkfs and dd are already a hardline floor.
export const SIMPLE_DENY_GLOBS = [
  'git *push*--force*',
  'git *push* -f',
  'git *push* -f *',
  'git *push* +*',
  'git *reset*--hard*',
  'diskutil *erase*',
  'diskutil zerodisk*',
  'diskutil randomdisk*'
]

const section = (config: HermesConfigRecord, key: string): HermesConfigRecord => {
  const value = config[key]

  return value && typeof value === 'object' && !Array.isArray(value) ? (value as HermesConfigRecord) : {}
}

/** The preset as a PUT body over the user's own config (no defaults). PUT
 *  /api/config deep-merges dicts but replaces lists, so `approvals.deny` is
 *  sent as the union with the user's globs; a value the user already set for
 *  checkpoints or the coding brief is left out, so it stays theirs. */
export function simpleAgentPreset(current: HermesConfigRecord): HermesConfigRecord {
  const deny = section(current, 'approvals').deny

  const preset: HermesConfigRecord = {
    approvals: { deny: [...new Set([...(Array.isArray(deny) ? deny : []), ...SIMPLE_DENY_GLOBS])] }
  }

  if (section(current, 'checkpoints').enabled == null) {
    preset.checkpoints = { enabled: true }
  }

  if (section(current, 'agent').coding_context == null) {
    preset.agent = { coding_context: 'off' }
  }

  return preset
}

/** Registry kind first, then the socket already dialed for the scope. Unknown
 *  is not local: a preset written to (or a terminal sign-in run for) a remote
 *  host would change someone else's machine. */
export function isLocalScope(scope: OnboardingScope): boolean {
  const kind = scope.connectionId ? registryConnectionKind(scope.connectionId) : undefined

  if (kind) {
    return kind === 'local'
  }

  return dialedGatewayModeFor(scope.connectionId ?? null, scope.profile || 'default') === 'local'
}

/**
 * Write the Simple preset into the profile onboarding is setting up. Local
 * connections only, and only from the first-run mode choice: it runs before
 * any session exists, so no prompt cache is invalidated. A failed read or
 * write is a toast, never a blocker — onboarding goes on either way.
 */
export async function applySimpleAgentDefaults(scope: OnboardingScope): Promise<void> {
  if (!isLocalScope(scope)) {
    // Said once, so a user who chose Simple knows the remote agent kept its own rules.
    notify({ kind: 'info', message: translateNow('onboarding.simpleDefaultsRemote') })

    return
  }

  const target = { connectionId: scope.connectionId ?? null, profile: scope.profile ?? null }

  try {
    const current = await getHermesConfigRecord(target, { includeDefaults: false })
    await saveHermesConfigRecord(simpleAgentPreset(current ?? {}), target)
  } catch (error) {
    notifyError(error, translateNow('onboarding.simpleDefaultsFailed'))
  }
}
