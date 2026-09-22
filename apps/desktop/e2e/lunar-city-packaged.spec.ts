import { execFileSync } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { _electron, expect, test } from '@playwright/test'

import { PACKAGED_BINARY_PATH } from './fixtures'
import {
  createLunarCityPopulationFixture,
  expectedStandardProjection,
  gpuPackagedLaunchOptions,
  readStandardPopulation,
  startPopulationGateways
} from './lunar-city-fixtures'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const OPT_IN = process.env.HERMES_LUNAR_CITY_PACKAGED_E2E === '1'

interface PerfRunnerModule {
  inspectPackagedTarget(input: { binaryPath: string; expectedGitSha: string }): {
    binaryPath: string
    buildStamp: { commit: string }
  }
  validateBridgeHandshake(
    handshake: unknown,
    expected: { buildSha: string; launchNonce: string; mainPid: number }
  ): Record<string, unknown>
}

interface PerfBridge {
  handshake(expected: { bridgeVersion: number; launchNonce: string }): unknown
  mountCity(): Promise<void>
  processMetrics(): Promise<{ gpuMemoryMiB: null | number; gpuMemorySource: string }>
  snapshot(): Promise<Record<string, unknown>>
}

async function runnerModule(): Promise<PerfRunnerModule> {
  const modulePath = '../scripts/perf/lunar-city-runner.mjs'

  return (await import(modulePath)) as PerfRunnerModule
}

for (const population of [25, 100, 250] as const) {
  test(`packaged bridge observes exact standard-adapter population ${population}`, async () => {
    test.slow()
    test.skip(!OPT_IN, 'Packaged Lunar City E2E is explicit opt-in and a skip is not acceptance evidence.')
    test.skip(!fs.existsSync(PACKAGED_BINARY_PATH), `Packaged binary is missing: ${PACKAGED_BINARY_PATH}`)

    const headSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' }).trim()
    const runner = await runnerModule()
    const target = runner.inspectPackagedTarget({ binaryPath: PACKAGED_BINARY_PATH, expectedGitSha: headSha })
    const fixture = createLunarCityPopulationFixture(population)
    let app: Awaited<ReturnType<typeof _electron.launch>> | undefined
    let gateways: Awaited<ReturnType<typeof startPopulationGateways>> | undefined

    try {
      gateways = await startPopulationGateways(fixture)
      const standardProjection = await readStandardPopulation(gateways)

      expect(standardProjection).toEqual(expectedStandardProjection(fixture.contract))
      test.skip(
        fixture.contract.entitiesByKind.subagent > 0,
        'No authenticated real-gateway fixture method emits subagent.start; packaged exact-population proof is blocked and this skip is not evidence.'
      )
      const nonce = `lunar-city-${randomUUID()}`

      const launch = gpuPackagedLaunchOptions({
        executablePath: target.binaryPath,
        userDataDir: fixture.userDataDir,
        env: {
          ...stringEnvironment(process.env),
          HERMES_HOME: fixture.hermesHome,
          HERMES_DESKTOP_APP_NAME: `Hermes Lunar City Packaged ${nonce}`,
          HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1',
          HERMES_LUNAR_CITY_PERF_ACCEPTANCE: '1',
          HERMES_LUNAR_CITY_PERF_NONCE: nonce
        }
      })

      expect(launch.args).not.toContain('--disable-gpu')
      app = await _electron.launch(launch)
      const page = await app.firstWindow()

      const bridgePresent = await page.evaluate(() =>
        Boolean((window as unknown as { __LUNAR_CITY_PERF__?: unknown }).__LUNAR_CITY_PERF__)
      )

      test.skip(!bridgePresent, 'Versioned packaged Lunar City bridge is absent; this run is not acceptance evidence.')

      const handshake = await page.evaluate(
        ({ launchNonce }) =>
          (window as unknown as { __LUNAR_CITY_PERF__: PerfBridge }).__LUNAR_CITY_PERF__.handshake({
            bridgeVersion: 1,
            launchNonce
          }),
        { launchNonce: nonce }
      )

      runner.validateBridgeHandshake(handshake, {
        buildSha: target.buildStamp.commit,
        launchNonce: nonce,
        mainPid: app.process().pid!
      })

      const result = await page.evaluate(async () => {
        const bridge = (window as unknown as { __LUNAR_CITY_PERF__: PerfBridge }).__LUNAR_CITY_PERF__
        await bridge.mountCity()

        return { gpu: await bridge.processMetrics(), snapshot: await bridge.snapshot() }
      })

      expect(result.gpu.gpuMemorySource).toMatch(/^(?:chromium-memory-infra-v1|unavailable)$/u)

      if (result.gpu.gpuMemorySource === 'chromium-memory-infra-v1') {
        expect(result.gpu.gpuMemoryMiB).toBeGreaterThanOrEqual(0)
      } else {
        expect(result.gpu.gpuMemoryMiB).toBeNull()
      }

      expect(result.snapshot).toMatchObject({
        population: {
          active: standardProjection.activity.active,
          lodMix: {
            far: fixture.contract.lod.far + fixture.contract.lod.aggregate,
            mid: fixture.contract.lod.mid,
            near: fixture.contract.lod.near
          },
          observed: population,
          source: 'lunar-city-snapshot-v1'
        },
        populationSourceMix: standardProjection.sourceMix
      })
    } finally {
      await app?.close().catch(() => undefined)
      await gateways?.close()
      fixture.cleanup()
    }
  })
}

function stringEnvironment(env: NodeJS.ProcessEnv): Record<string, string> {
  return Object.fromEntries(
    Object.entries(env).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
  )
}
