// `desktop.ssh_connect_timeout` / `desktop.ssh_forward_timeout` (#132508).
//
// Desktop gives ssh 15s to connect and 15s to set up its port forward. The
// connect limit goes to ssh on the command line, so the user's ~/.ssh/config
// cannot raise it; these two config.yaml keys are the only knob. They are read
// once, before the first window, from the same file text as the launch keys.

import { resolveSshBudgetMs } from './ssh-connection'

/** Budgets to hand to every desktop SshConnection; an unset one keeps the default. */
export interface DesktopSshBudgets {
  connectTimeoutMs?: number
  forwardTimeoutMs?: number
}

const KEYS = {
  ssh_connect_timeout: 'connectTimeoutMs',
  ssh_forward_timeout: 'forwardTimeoutMs'
} as const

/** The raw values of the two keys under a top-level `desktop:` mapping. */
function rawDesktopSshTimeouts(yamlText: string): Partial<Record<keyof typeof KEYS, string>> {
  const lines = String(yamlText ?? '').split(/\r?\n/)
  const start = lines.findIndex(line => /^desktop:\s*(#.*)?$/.test(line))
  const raw: Partial<Record<keyof typeof KEYS, string>> = {}

  for (let i = start + 1; start >= 0 && i < lines.length && !/^[^\s#]/.test(lines[i]); i += 1) {
    const keyed = /^ {2}(ssh_connect_timeout|ssh_forward_timeout):\s*(.*?)\s*$/.exec(lines[i])
    const value = keyed?.[2].replace(/\s+#.*$/, '').trim().replace(/^(['"])(.*)\1$/, '$2')

    if (keyed && value) {
      raw[keyed[1] as keyof typeof KEYS] = value
    }
  }

  return raw
}

/**
 * Read both budgets out of `config.yaml` text. Same narrow shape as the launch
 * keys: two-space-indented under a top-level `desktop:`. A value that is not a
 * whole number of seconds is ignored with one warning, so a typo keeps the
 * default instead of shortening a limit that was already too short.
 */
export function readDesktopSshBudgets(
  yamlText: string,
  log: Pick<Console, 'log' | 'warn'> = console
): DesktopSshBudgets {
  const budgets: DesktopSshBudgets = {}

  for (const [key, value] of Object.entries(rawDesktopSshTimeouts(yamlText))) {
    const ms = resolveSshBudgetMs(value)

    if (ms === undefined) {
      log.warn(
        `[hermes] config.yaml: desktop.${key} was ignored. It takes a whole number of seconds (up to 120); keeping 15.`
      )
    } else {
      budgets[KEYS[key as keyof typeof KEYS]] = ms
      log.log(`[hermes] desktop.${key} from config.yaml: ${ms / 1000}s`)
    }
  }

  return budgets
}
