import os from 'node:os'
import path from 'node:path'

import { execText } from './backend-claim'
import { parseKnownHostsFingerprints } from './ssh-connection'

export interface ManagedRolloutSshConfig {
  host: string
  user?: string
  port?: number
  keyPath?: string
}

const MAX_KNOWN_HOSTS_FILES = 4
const MAX_CONCURRENT_LOOKUPS = 4
const HOST_KEY_LOOKUP_TIMEOUT_MS = 5_000

/** Read the accepted OpenSSH host key from the configured known-hosts files. */
export async function readVerifiedHostKeyFingerprint(config: ManagedRolloutSshConfig): Promise<string> {
  const sshBinary = process.platform === 'win32'
    ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')
    : 'ssh'

  const target = config.user ? `${config.user}@${config.host}` : config.host
  const args = ['-G']

  if (config.port) {args.push('-p', String(config.port))}

  if (config.keyPath) {args.push('-i', config.keyPath)}
  args.push('--', target)

  const expanded = await execText(sshBinary, args, { timeout: 10_000 })
  const knownHosts = new Set<string>()
  let resolvedHost = String(config.host)
  let resolvedPort = Number(config.port || 22)

  for (const line of String(expanded || '').split(/\r?\n/)) {
    const fields = line.trim().split(/\s+/)

    if (fields[0] === 'hostname' && fields[1]) {resolvedHost = fields[1]}

    if (fields[0] === 'port' && /^\d+$/.test(fields[1] || '')) {resolvedPort = Number(fields[1])}

    if (fields.length < 2 || !['userknownhostsfile', 'globalknownhostsfile'].includes(fields[0])) {continue}

    for (const candidate of fields.slice(1)) {
      if (!candidate || candidate === 'none' || candidate.includes('%')) {continue}

      const resolved = candidate.startsWith('~/')
        ? path.join(os.homedir(), candidate.slice(2))
        : path.resolve(candidate)

      knownHosts.add(resolved)

      if (knownHosts.size > MAX_KNOWN_HOSTS_FILES) {
        throw new Error('Managed rollout host-key lookup exceeds the known-hosts file limit.')
      }
    }
  }

  const hostNames = new Set<string>([String(config.host), resolvedHost])
  const ports = new Set<number>([Number(config.port || 22), resolvedPort])

  for (const port of ports) {
    if (port !== 22) {
      hostNames.add(`[${resolvedHost}]:${port}`)
      hostNames.add(`[${config.host}]:${port}`)
    }
  }

  const lookups: Array<{ file: string; host: string }> = []

  for (const knownHostsFile of knownHosts) {
    for (const hostName of hostNames) {
      lookups.push({ file: knownHostsFile, host: hostName })
    }
  }

  const outputs: string[] = []

  for (let offset = 0; offset < lookups.length; offset += MAX_CONCURRENT_LOOKUPS) {
    const batch = lookups.slice(offset, offset + MAX_CONCURRENT_LOOKUPS)

    const results = await Promise.all(batch.map(async ({ file, host }) => {
      try {
        return await execText('ssh-keygen', ['-F', host, '-f', file], { timeout: HOST_KEY_LOOKUP_TIMEOUT_MS })
      } catch (error) {
        // ssh-keygen exits 1 for no matching host. Missing or unreadable files,
        // timeouts, and other lookup failures leave identity unverified.
        if ((error as { code?: unknown }).code === 1) {return ''}

        throw error
      }
    }))

    outputs.push(...results)
  }

  const fingerprints = parseKnownHostsFingerprints(outputs)

  if (fingerprints.length !== 1) {
    throw new Error(`Managed rollout host-key evidence is ambiguous (${fingerprints.length} fingerprints).`)
  }

  return fingerprints[0]
}
