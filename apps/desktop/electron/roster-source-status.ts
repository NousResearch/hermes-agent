export function rosterSourceStatus(source: { profiles: string[] | null; error?: string; needsSignIn?: boolean }) {
  return {
    reachable: source.profiles !== null && (!source.error || source.error === 'connect-on-demand'),
    ...(source.error ? { error: source.error } : {}),
    ...(source.needsSignIn ? { needsSignIn: true } : {})
  }
}
