/**
 * update-channel-record.ts
 *
 * The install-scoped update channel record, shared with the CLI updater.
 *
 * Contract (same as hermes_cli/update_channel.py):
 *   - One JSON record per INSTALLATION at
 *     `<default hermes root>/update-channel.json`, not per profile and not
 *     Electron userData. The CLI resolves the identical path via
 *     get_default_hermes_root(); normalizeHermesHomeRoot() mirrors that
 *     profile-unwrapping, so both surfaces read/write one file.
 *   - Schema: {"schema_version": 1, "channel": "stable" | "beta"}.
 *   - Missing or malformed reads as "stable" and a read NEVER writes.
 *   - Only an explicit selection (the About selector or
 *     `hermes update --channel ...`) persists the record.
 *   - "beta" means tracking the moving main branch — not a release candidate.
 */

import fs from 'node:fs'
import path from 'node:path'

import { normalizeHermesHomeRoot } from './backend-env'

export const UPDATE_CHANNEL_RECORD_FILENAME = 'update-channel.json'
export const UPDATE_CHANNEL_SCHEMA_VERSION = 1

export type HermesUpdateChannel = 'stable' | 'beta'

export interface UpdateChannelRecord {
  schemaVersion: typeof UPDATE_CHANNEL_SCHEMA_VERSION
  channel: HermesUpdateChannel
}

export function updateChannelRecordPath(
  hermesHome: string,
  { pathModule = path }: { pathModule?: typeof path } = {}
): string {
  const root = normalizeHermesHomeRoot(hermesHome, { pathModule })
  return pathModule.join(root, UPDATE_CHANNEL_RECORD_FILENAME)
}

export function parseUpdateChannelRecord(raw: string): UpdateChannelRecord | null {
  try {
    const parsed = JSON.parse(raw)
    const channel = parsed?.channel

    if (channel === 'stable' || channel === 'beta') {
      return { schemaVersion: UPDATE_CHANNEL_SCHEMA_VERSION, channel }
    }

    return null
  } catch {
    return null
  }
}

export function readUpdateChannel(
  hermesHome: string,
  {
    readFileSync = (p: string) => fs.readFileSync(p, 'utf8'),
    pathModule = path
  }: {
    readFileSync?: (p: string) => string
    pathModule?: typeof path
  } = {}
): HermesUpdateChannel {
  try {
    const record = parseUpdateChannelRecord(
      readFileSync(updateChannelRecordPath(hermesHome, { pathModule }))
    )

    // Missing or invalid record: stable, and NEVER written by a read.
    return record ? record.channel : 'stable'
  } catch {
    return 'stable'
  }
}

export function writeUpdateChannel(
  hermesHome: string,
  channel: HermesUpdateChannel,
  {
    writeFileSync = (p: string, data: string) => fs.writeFileSync(p, data, 'utf8'),
    pathModule = path
  }: {
    writeFileSync?: (p: string, data: string) => void
    pathModule?: typeof path
  } = {}
): UpdateChannelRecord {
  const record: UpdateChannelRecord = {
    schemaVersion: UPDATE_CHANNEL_SCHEMA_VERSION,
    channel: channel === 'beta' ? 'beta' : 'stable'
  }

  const target = updateChannelRecordPath(hermesHome, { pathModule })
  const temporary = `${target}.tmp`

  // Persist in the CLI's schema (snake_case keys): hermes_cli/update_channel.py
  // reads this exact file; a camelCase key would read as invalid → stable.
  const payload = { schema_version: UPDATE_CHANNEL_SCHEMA_VERSION, channel: record.channel }

  fs.mkdirSync(pathModule.dirname(target), { recursive: true })
  writeFileSync(temporary, `${JSON.stringify(payload, null, 2)}\n`)
  fs.renameSync(temporary, target)

  return record
}
