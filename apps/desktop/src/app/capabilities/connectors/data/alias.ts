import type { AccountRow } from '../types'

/** The gateway's account-name format (`ConnectorAlias` in the connector contracts). */
const ACCOUNT_ALIAS = /^[a-z0-9][a-z0-9-]{0,31}$/

export type AliasProblem = 'invalid' | 'taken'

/** Why `alias` cannot name an account among `accounts`, or null when it can. `self` is the account being renamed. */
export function aliasProblem(alias: string, accounts: readonly AccountRow[], self?: string): AliasProblem | null {
  if (!ACCOUNT_ALIAS.test(alias)) {
    return 'invalid'
  }

  return accounts.some(account => account.alias === alias && account.connection_id !== self) ? 'taken' : null
}
