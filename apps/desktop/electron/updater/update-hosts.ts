import { readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

/**
 * Hosts this install's update reaches beyond the public ones in UPDATE_HOSTS: the
 * checkout's origin (a fork, an internal git mirror) and every package index or npm
 * registry the user configured (#123387 review S2). The system proxy is asked about each
 * one, so a mirror the system sends direct joins NO_PROXY instead of being proxied.
 *
 * Only http(s) URLs count: git over ssh and file:// remotes never read proxy env.
 */

// uv and pip index settings (any letter case; UV_INDEX_<NAME>_URL too) and npm's registry.
const INDEX_KEYS = [
  'uv_index_url',
  'uv_extra_index_url',
  'uv_default_index',
  'uv_index',
  'pip_index_url',
  'pip_extra_index_url',
  'npm_config_registry'
]

function httpHost(value: string): string | null {
  // UV_INDEX entries may be `name=url`.
  const url = value.includes('=') && !/^https?:/i.test(value) ? value.slice(value.indexOf('=') + 1) : value

  try {
    const parsed = new URL(url.trim())

    return /^https?:$/.test(parsed.protocol) && parsed.hostname ? parsed.hostname.toLowerCase() : null
  } catch {
    return null
  }
}

function readText(file: string): string {
  try {
    return readFileSync(file, 'utf8')
  } catch {
    return ''
  }
}

/** `key = value` lines of one INI-style file, `[section]` names lower-cased; '' before any section. */
function iniValues(text: string, section: string, keys: string[]): string[] {
  let current = ''
  const values: string[] = []

  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    const header = /^\[(.+)\]$/.exec(line)

    if (header) {
      current = header[1].trim().toLowerCase()
    } else if (current === section) {
      const [key, ...rest] = line.split('=')

      if (rest.length && keys.includes(key.trim().toLowerCase())) {
        values.push(
          rest
            .join('=')
            .trim()
            .replace(/^["']|["']$/g, '')
        )
      }
    }
  }

  return values
}

function pipConfigFiles(env: NodeJS.ProcessEnv, home: string): string[] {
  const explicit = env.PIP_CONFIG_FILE

  if (process.platform === 'win32') {
    return [
      path.join(home, 'pip', 'pip.ini'),
      path.join(env.APPDATA || path.join(home, 'AppData', 'Roaming'), 'pip', 'pip.ini'),
      ...(explicit ? [explicit] : [])
    ]
  }

  return [
    path.join(home, '.pip', 'pip.conf'),
    path.join(home, 'Library', 'Application Support', 'pip', 'pip.conf'),
    path.join(env.XDG_CONFIG_HOME || path.join(home, '.config'), 'pip', 'pip.conf'),
    ...(explicit ? [explicit] : [])
  ]
}

export function configuredUpdateHosts(
  updateRoot: string,
  env: NodeJS.ProcessEnv,
  home: string = os.homedir()
): string[] {
  const values: string[] = iniValues(readText(path.join(updateRoot, '.git', 'config')), 'remote "origin"', ['url'])

  for (const [key, value] of Object.entries(env)) {
    const name = key.toLowerCase()

    if (value && (INDEX_KEYS.includes(name) || /^uv_index_.+_url$/.test(name))) {
      values.push(...value.split(/\s+/))
    }
  }

  const npmrc = env.npm_config_userconfig || env.NPM_CONFIG_USERCONFIG || path.join(home, '.npmrc')
  values.push(...iniValues(readText(npmrc), '', ['registry']))

  for (const file of pipConfigFiles(env, home)) {
    values.push(...iniValues(readText(file), 'global', ['index-url', 'extra-index-url']).flatMap(v => v.split(/\s+/)))
  }

  return [...new Set(values.map(httpHost).filter((host): host is string => Boolean(host)))]
}
