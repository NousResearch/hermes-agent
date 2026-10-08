import crypto from 'node:crypto'

export function browserArguments(action: string, args: Record<string, unknown>, ownerScope: string) {
  if ('session' in args) throw new Error('Desktop owns the browser session label.')
  if (action !== 'get_browser_state' && !action.startsWith('browser_')) return args
  return { ...args, session: 'hermes-' + crypto.createHash('sha256').update(ownerScope).digest('hex').slice(0, 24) }
}
