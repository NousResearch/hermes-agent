import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

// main.ts has no unit harness. launched-profile.test.ts covers the functions that keep a profile a
// plugin runs off this computer's own backend; this file covers that main.ts calls each of them at
// the place where it matters. Each case reads the source and fails when the call is removed or no
// longer stands before the start it guards.
const source = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')

describe('main.ts calls the launched-profile guards where a profile could start or answer locally', () => {
  it('refuses a marked profile before a pooled dial starts a backend for it', () => {
    expect(source).toMatch(
      /launchedProfiles\.assertMayRunLocally\(profile\)\s+const connection = await ensureLocalGateway\(async \(\) => \{\s+const backend = await ensureRuntime\(\s+await resolveHermesBackend\(\['--profile', profile, 'gateway', 'ensure', '--json'\]\)/
    )
  })

  it('refuses a marked profile before the shared host backend serves it by scope', () => {
    expect(source).toMatch(
      /if \(route\.backend === 'primary'\) \{\s+launchedProfiles\.assertScopeMayRunLocally\(key, routeOpts\)\s+const joined = await joinPrimaryBoot\(/
    )
  })

  it('refuses a marked profile before the primary backend starts for it', () => {
    expect(source).toMatch(
      /prepareLocalBackend: async \(\) => \{\s+launchedProfiles\.assertMayRunLocally\(activeProfile \|\| primaryProfile\)[\s\S]{0,200}?return resolveHermesBackend\(backendArgs/
    )
  })

  it('tells the route resolver when a plugin runs the primary, so other local profiles get a backend of their own', () => {
    expect(source).toMatch(
      /primaryRemoteActive: primaryBackendIsRemote\(\),[\s\S]{0,200}?primaryLaunched: profileIsLaunched\(config, primaryProfileKey\(\)\),/
    )
  })

  it('tells the plugin-root copy which homes a sandbox can write', () => {
    expect(source).toMatch(
      /registerFsIpc\(\{\s+hermesHome: HERMES_HOME,\s+isLaunchedProfile: launchedProfiles\.homeIsSandboxed,/
    )
  })

  it("answers a plugin-run profile's listing itself before the request is forwarded anywhere", () => {
    expect(source).toMatch(
      /const launchedAnswer = await launchedProfiles\.answer\([\s\S]{0,400}?\)\s+if \(launchedAnswer !== undefined\) \{\s+return launchedAnswer\s+\}\s+if \(registryConnectionId\) \{/
    )
  })
})
