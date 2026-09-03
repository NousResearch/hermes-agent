import path from 'node:path'

/**
 * Resolve internal perf-only renderer modules at Vite config time. Normal
 * shipped production must not include the fixture implementation; the perf
 * harness explicitly opts its minified renderer back in with VITE_PERF_PROBE.
 */
export function sessionOpenPerfFixtureEntry(root, command, env) {
  const filename =
    command === 'serve' || env.VITE_PERF_PROBE === '1'
      ? 'session-open-perf-fixture.ts'
      : 'session-open-perf-fixture.noop.ts'

  return path.resolve(root, 'src/app/session/hooks/use-session-actions', filename)
}
