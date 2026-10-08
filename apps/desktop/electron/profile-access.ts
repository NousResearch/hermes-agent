export function requestedProfileScope(action: string, args: Record<string, unknown>, boundScope?: string) {
  if (action !== 'browser_prepare') return boundScope
  const mode = (args.profile as { mode?: string } | undefined)?.mode
  if (mode === 'athena_profile') return `athena_profile:${args.browser || 'chrome'}`
  if (mode === 'existing_profile') return `existing_profile:${args.pid}:${args.window_id}`
  return undefined
}

export function needsProfileApproval(grants: Set<string>, scope?: string) { return Boolean(scope && !grants.has(scope)) }
