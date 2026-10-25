import { parseRemoteProfileListing } from './connection-registry'
import { assertWindowsRemoteInstallUpdateClear, helper } from './windows-remote-lifecycle'
import type { WindowsRuntime } from './windows-runtime-command'

// The runtime helper reads through the install's own root and profile predicate;
// it opens no configs, credentials or sessions and never creates a home or identity.
async function readInventory(ssh: any, runtime: WindowsRuntime): Promise<any> {
  await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome, runtime.python)

  const parsed = await helper(ssh, runtime, 'inventory')

  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Windows inventory did not return valid JSON.')
  }

  return parsed
}

export async function listWindowsRemoteProfiles(ssh: any, runtime: WindowsRuntime): Promise<string[]> {
  const result = await readInventory(ssh, runtime)

  if (
    !Array.isArray(result.profiles) ||
    !result.profiles.every((name: unknown) => typeof name === 'string') ||
    !result.profiles.includes('default')
  ) {
    throw new Error('Windows inventory did not return canonical profile names.')
  }

  return parseRemoteProfileListing(result.profiles.join('\n'))
}

export async function readWindowsRemoteInstallId(ssh: any, runtime: WindowsRuntime): Promise<string | undefined> {
  const result = await readInventory(ssh, runtime)

  if (!Object.hasOwn(result, 'installId')) {
    throw new Error('Windows inventory did not report its install identity.')
  }

  const id = typeof result.installId === 'string' ? result.installId.trim().toLowerCase() : ''

  return /^[0-9a-f]{32}$/.test(id) ? id : undefined
}
