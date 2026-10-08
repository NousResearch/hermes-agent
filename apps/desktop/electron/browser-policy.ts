export function validateBrowserPrepare(args: Record<string, unknown>) {
  if (args.target_id !== undefined || args.tab_id !== undefined) throw new Error('Preparation cannot retarget an existing browser capability.')
  if (args.browser !== undefined && !['chrome', 'edge', 'firefox', 'safari'].includes(String(args.browser))) throw new Error('Unsupported browser product.')
  const profile = args.profile as { mode?: string } | undefined
  if (args.browser === 'safari' && (process.platform !== 'darwin' || profile?.mode !== 'isolated_new')) throw new Error('Safari requires macOS and isolated_new mode. Personal Safari tabs and persistent profiles are not supported.')
  if (profile?.mode === 'existing_profile') {
    if (args.browser === 'firefox') throw new Error('Existing Firefox attachment is not supported yet. Use its private or Athena profile mode.')
    if (args.strategy || args.allow_launch !== false || !Number.isSafeInteger(args.pid) || Number(args.pid) < 1 || !Number.isSafeInteger(args.window_id) || Number(args.window_id) < 1) throw new Error('Existing browser mode requires exact positive pid/window_id and allow_launch=false.')
    return
  }
  if (profile?.mode === 'athena_profile' && !args.strategy && args.pid === undefined && args.window_id === undefined && args.allow_launch === true) return
  if (args.strategy || args.pid !== undefined || args.window_id !== undefined || args.allow_launch !== true || !['isolated_new', 'isolated_named'].includes(profile?.mode || '')) {
    throw new Error('This browser prototype prepares a separate driver-owned profile. Existing personal profile attachment is not enabled.')
  }
}
