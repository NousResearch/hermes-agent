/** Remove launch overrides after the shared fixture strips provider credentials. */
export function stripKoreanQaOverrides(env: NodeJS.ProcessEnv): void {
  for (const k of Object.keys(env)) {
    if (
      (k !== 'MOCK_API_KEY' && /TOKEN|PASSWORD|SECRET|API_KEY|PAT_|CREDENTIAL|ACCESS_KEY|PRIVATE_KEY/i.test(k)) ||
      [
        'HERMES_DESKTOP_DEV_SERVER',
        'HERMES_DESKTOP_REMOTE_URL',
        'HERMES_DESKTOP_FAKE_BOOT',
        'ELECTRON_RUN_AS_NODE',
        'NODE_OPTIONS'
      ].includes(k) ||
      k.startsWith('HERMES_DESKTOP_BOOT_FAKE')
    ) {
      delete env[k]
    }
  }
}
