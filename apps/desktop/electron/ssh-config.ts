/**
 * ssh-config.ts
 *
 * Pure, electron-free helpers for reading the user's OpenSSH client config:
 * `Host` aliases for the settings UI's suggestions, `Include` traversal
 * (read-only), and `ssh -G` output parsing. No `import 'electron'` so it's
 * unit-testable without Electron; main.ts wires the fs + `ssh -G` exec in.
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

function parseSshConfigHosts(text) {
  const hosts: string[] = []
  const seen = new Set()

  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.trim()

    if (!line || line.startsWith('#')) {
      continue
    }

    const m = /^host\s+(.+)$/i.exec(line)

    if (!m) {
      continue
    }

    for (const pattern of m[1].split(/\s+/)) {
      if (!pattern || pattern.includes('*') || pattern.includes('?') || pattern.startsWith('!')) {
        continue
      }

      if (!seen.has(pattern)) {
        seen.add(pattern)
        hosts.push(pattern)
      }
    }
  }

  return hosts
}

function parseSshConfigIncludes(text) {
  const includes: string[] = []

  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.trim()

    if (!line || line.startsWith('#')) {
      continue
    }

    const m = /^include\s+(.+)$/i.exec(line)

    if (!m) {
      continue
    }

    for (const token of m[1].split(/\s+/)) {
      if (token) {
        includes.push(token)
      }
    }
  }

  return includes
}

function collectSshConfigHosts(rootPath = '', deps: any = {}) {
  const readFile =
    deps.readFile ||
    (p => {
      try {
        return fs.readFileSync(p, 'utf8')
      } catch {
        return null
      }
    })

  const homeDir = deps.homeDir || os.homedir()
  const root = rootPath || path.join(homeDir, '.ssh', 'config')
  const sshDir = path.join(homeDir, '.ssh')

  const out: string[] = []
  const seen = new Set()
  const visited = new Set()

  const resolveIncludePath = token => {
    if (token.startsWith('~/')) {
      return path.join(homeDir, token.slice(2))
    }

    if (path.isAbsolute(token)) {
      return token
    }

    return path.join(sshDir, token)
  }

  const walk = (filePath, depth) => {
    if (depth > 8 || visited.has(filePath)) {
      return
    }

    visited.add(filePath)
    const text = readFile(filePath)

    if (text == null) {
      return
    }

    for (const host of parseSshConfigHosts(text)) {
      if (!seen.has(host)) {
        seen.add(host)
        out.push(host)
      }
    }

    for (const token of parseSshConfigIncludes(text)) {
      const target = resolveIncludePath(token)
      const expanded = deps.globSync ? deps.globSync(target) : [target]

      for (const p of expanded) {
        walk(p, depth + 1)
      }
    }
  }

  walk(root, 0)

  return out
}

function parseSshGOutput(text) {
  const out: { hostname: string | null; user: string | null; port: number | null; identityFile: string | null } = {
    hostname: null,
    user: null,
    port: null,
    identityFile: null
  }

  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.trim()

    if (!line) {
      continue
    }

    const sp = line.indexOf(' ')

    if (sp === -1) {
      continue
    }

    const key = line.slice(0, sp).toLowerCase()
    const value = line.slice(sp + 1).trim()

    if (key === 'hostname' && !out.hostname) {
      out.hostname = value
    } else if (key === 'user' && !out.user) {
      out.user = value
    } else if (key === 'port' && !out.port) {
      out.port = Number.parseInt(value, 10) || null
    } else if (key === 'identityfile' && !out.identityFile) {
      out.identityFile = value
    }
  }

  return out
}

export { collectSshConfigHosts, parseSshConfigHosts, parseSshConfigIncludes, parseSshGOutput }
