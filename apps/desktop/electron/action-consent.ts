export type ActionConsentScope = 'background' | 'foreground'

export function actionConsentScope(action: string, args: Record<string, unknown>): ActionConsentScope {
  const target = args.target as { kind?: string } | undefined
  const profile = args.profile as { mode?: string } | undefined
  return action === 'bring_to_front' || args.browser === 'safari' || (typeof args.target_id === 'string' && args.target_id.startsWith('saf-')) || (action === 'browser_prepare' && profile?.mode === 'existing_profile') || args.delivery_mode === 'foreground' || args.bring_to_front === true || args.scope === 'desktop' || target?.kind === 'desktop'
    ? 'foreground' : 'background'
}

export function consentCovers(grants: Set<ActionConsentScope>, action: string, args: Record<string, unknown>) {
  return grants.has(actionConsentScope(action, args))
}
