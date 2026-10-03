export function retainBrowserAfterError(action: string, args: Record<string, unknown>, message: string) {
  // Keep an already-scoped adapter available for inspection; never replay input.
  // Preparation failures and privacy-boundary failures still revoke the session.
  return action !== 'browser_prepare'
    && (action === 'get_browser_state' || action.startsWith('browser_'))
    && typeof args.target_id === 'string' && args.target_id.startsWith('obt-')
    && !message.includes('Private Edge profile enabled sync')
}
