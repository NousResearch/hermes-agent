import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { load } from 'js-yaml'
import { expect, test } from 'vitest'

const root = fileURLToPath(new URL('../', import.meta.url))
const workflow = load(readFileSync(path.join(root, '.github/workflows/desktop-bundled-release.yml'), 'utf8'))
const tag = 'channel.20261028T120000Z-v1.2.3'
const token = 'ab12cd34ef56ab78'
const request = {
  schema: 1, buildId: 'a'.repeat(32), channel: 'feed-preview', sequence: 1,
  repository: 'fixture/project', commit: 'b'.repeat(40), sourceVersion: '1.2.3',
  releaseTag: tag, version: '1.2.3+channel.20261028T120000Z', windowsVersion: '26.1028.12.0',
  publicBase: 'https://builds.example.test', bundleEnv: {},
  identity: { token, displayName: 'Hermes feed-preview', appId: `com.nousresearch.hermes-channel-${token}`,
    appNamePascal: `HermesChannel${token}`, artifactNamePascal: 'HermesBundled',
    cliName: 'hermes-feed-preview', windowsExecutableName: 'hermes-feed-preview',
    msixAppIdWithOrg: `NousResearch.HermesChannel${token}` },
}

for (const arch of ['arm64', 'x64']) {
  test.each([
    ['stable', 'v1.2.3'], ['canary', 'canary.20261028T120000Z-v1.2.3'], ['channel', tag],
  ])(`${arch} stages the actual %s packaging feed under its architecture`, (policy, payloadTag) => {
    const env = { ...process.env, HERMES_DESKTOP_VARIANT: 'bundled', HERMES_BUILD_COMMIT: '',
      HERMES_PAYLOAD_TAG: payloadTag, HERMES_PAYLOAD_VERSION: '', CLOUDFLARE_R2_PUBLIC_URL: request.publicBase,
      _HERMES_CHANNEL_REQUEST_JSON: policy === 'channel' ? JSON.stringify(request) : '' }
    const publish = JSON.parse(execFileSync(process.execPath, ['-e',
      "console.log(JSON.stringify(require('./apps/desktop/electron-builder.config.cjs').mac.publish))"],
    { cwd: root, env, encoding: 'utf8' }))
    const feedChannel = publish[0].channel
    const leg = Object.values(workflow.jobs).find(job =>
      job.strategy?.matrix?.target?.some(target => target.label === `darwin-${arch}`)
      && job.steps?.some(step => step.name === 'Rename the feed yml per arch'))
    const step = leg.steps.find(step => step.name === 'Rename the feed yml per arch')
    const script = step.run.replaceAll('${{ matrix.target.label }}', `darwin-${arch}`)
      .replaceAll('${{ needs.validate.outputs.channel }}', policy)
    const temp = mkdtempSync(path.join(os.tmpdir(), 'channel-feed-staging-'))
    try {
      const out = path.join(temp, 'apps/desktop/release')
      mkdirSync(out, { recursive: true })
      const body = 'version: 1.2.3\nfiles: []\n'
      writeFileSync(path.join(out, `${feedChannel}-mac.yml`), body)
      const result = spawnSync('bash', ['-e', '-o', 'pipefail', '-c', script], { cwd: temp, encoding: 'utf8' })
      expect(result.status, result.stderr || String(result.error || '')).toBe(0)
      expect(readdirSync(out)).toEqual([`${arch}-${feedChannel}-mac.yml`])
      expect(readFileSync(path.join(out, `${arch}-${feedChannel}-mac.yml`), 'utf8')).toBe(body)
    } finally {
      rmSync(temp, { recursive: true, force: true })
    }
  })
}
