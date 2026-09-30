import assert from 'node:assert/strict'
import { once } from 'node:events'
import fs from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import https from 'node:https'
import type { AddressInfo } from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import {
  buildPinArgs,
  buildPosixPinArgs,
  cachedScriptPath,
  cleanInstallerLogLine,
  downloadInstallScript,
  hasExistingGitCheckout,
  installRefForStamp,
  isPinnedCommit,
  prepareCachedScriptBytes,
  resolveInstallScript,
  resolveMarkerPinnedCommit,
  runBootstrap,
  scriptUrls
} from './bootstrap-runner'

const SCRIPT_NAME = process.platform === 'win32' ? 'install.ps1' : 'install.sh'
const ZERO_COMMIT = '0000000000000000000000000000000000000000'
const DOWNLOAD_BOM = process.platform === 'win32' ? '\uFEFF' : ''

function mkTmpHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-test-'))
}

test('runBootstrap bails immediately when the signal is already aborted', async () => {
  const controller = new AbortController()
  controller.abort()

  const events = []

  const result = await runBootstrap({
    installStamp: null,
    activeRoot: '/tmp/hermes-runner-test',
    sourceRepoRoot: null,
    hermesHome: '/tmp/hermes-runner-test',
    logRoot: '/tmp/hermes-runner-test',
    onEvent: ev => events.push(ev),
    abortSignal: controller.signal
  })

  // Cancelled before any install script is spawned.
  assert.deepEqual(result, { ok: false, cancelled: true })
  assert.ok(
    events.some(ev => ev.type === 'failed' && /cancelled/i.test(ev.error)),
    'should emit a cancelled failure event'
  )
})

test('existing checkout detection requires git metadata', () => {
  const home = mkTmpHome()

  try {
    const activeRoot = path.join(home, 'hermes-agent')
    assert.equal(hasExistingGitCheckout(activeRoot), false)

    fs.mkdirSync(path.join(activeRoot, '.git'), { recursive: true })
    assert.equal(hasExistingGitCheckout(activeRoot), true)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('fresh bootstrap args include the packaged commit pin', () => {
  const installStamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(buildPinArgs(installStamp), ['-Commit', installStamp.commit, '-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp,
      activeRoot: '/tmp/hermes-agent',
      hermesHome: '/tmp/hermes'
    }),
    ['--dir', '/tmp/hermes-agent', '--hermes-home', '/tmp/hermes', '--branch', 'main', '--commit', installStamp.commit]
  )
})

test('existing-checkout bootstrap args keep branch but skip the packaged commit pin', () => {
  const installStamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(buildPinArgs(installStamp, { pinCommit: false }), ['-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp,
      activeRoot: '/tmp/hermes-agent',
      hermesHome: '/tmp/hermes',
      pinCommit: false
    }),
    ['--dir', '/tmp/hermes-agent', '--hermes-home', '/tmp/hermes', '--branch', 'main']
  )
})

test('fallback install stamps use an unpinned branch ref', () => {
  const stamp = { commit: ZERO_COMMIT, branch: 'main' }

  assert.equal(isPinnedCommit(ZERO_COMMIT), false)
  assert.deepEqual(installRefForStamp(stamp), {
    ref: 'main',
    cacheKey: 'branch-main',
    pinned: false
  })
  // Must NOT pass -Commit / --commit for the all-zero placeholder.
  assert.deepEqual(buildPinArgs(stamp), ['-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp: stamp,
      activeRoot: '/tmp/hermes',
      hermesHome: '/tmp/home'
    }),
    ['--dir', '/tmp/hermes', '--hermes-home', '/tmp/home', '--branch', 'main']
  )
})

test('existing-checkout installer ref follows the branch instead of the packaged commit', () => {
  const stamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(installRefForStamp(stamp, { pinCommit: false }), {
    ref: 'main',
    cacheKey: 'branch-main',
    pinned: false
  })
  assert.deepEqual(installRefForStamp(stamp), {
    ref: stamp.commit,
    cacheKey: stamp.commit,
    pinned: true
  })
})

test('resolveMarkerPinnedCommit prefers installed checkout HEAD over the packaged artifact', () => {
  const realHead = 'c'.repeat(40)
  assert.equal(
    resolveMarkerPinnedCommit({ commit: ZERO_COMMIT, branch: 'main' }, '/tmp/checkout', {
      resolveHead: () => realHead
    }),
    realHead
  )
  assert.equal(
    resolveMarkerPinnedCommit({ commit: 'd'.repeat(40), branch: 'main' }, '/tmp/checkout', {
      resolveHead: () => realHead
    }),
    realHead,
    'the installed checkout owns source runtime identity'
  )
  assert.equal(
    resolveMarkerPinnedCommit({ commit: ZERO_COMMIT, branch: 'main' }, '/tmp/missing', {
      resolveHead: () => null
    }),
    null
  )
})

test('resolveInstallScript downloads fallback stamps by branch instead of zero commit', async () => {
  const home = mkTmpHome()

  try {
    const cached = cachedScriptPath(home, 'branch-main')
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'stale branch installer\n')

    const refs = []

    const result = await resolveInstallScript({
      installStamp: { commit: ZERO_COMMIT, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      _download: async (ref, destPath) => {
        refs.push(ref)
        fs.mkdirSync(path.dirname(destPath), { recursive: true })
        fs.writeFileSync(destPath, '#!/bin/sh\necho fallback branch\n')

        return destPath
      }
    })

    assert.deepEqual(refs, ['main'])
    assert.equal(result.source, 'download')
    assert.equal(result.commit, null)
    assert.notEqual(result.path, cached)
    assert.equal(fs.readFileSync(cached, 'utf8'), 'stale branch installer\n')
    assert.match(fs.readFileSync(result.path, 'utf8'), /fallback branch/)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript refreshes the live branch for an existing checkout', async () => {
  const home = mkTmpHome()

  try {
    const commit = 'a'.repeat(40)
    const cached = cachedScriptPath(home, 'branch-main')
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'stale installer\n')

    const refs = []

    const result = await resolveInstallScript({
      installStamp: { commit, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      pinCommit: false,
      _download: async (ref, destPath) => {
        refs.push(ref)
        fs.writeFileSync(destPath, 'fresh branch installer\n')

        return destPath
      }
    })

    assert.deepEqual(refs, ['main'])
    assert.equal(result.source, 'download')
    assert.equal(result.commit, null)
    assert.notEqual(result.path, cached)
    assert.equal(fs.readFileSync(cached, 'utf8'), 'stale installer\n')
    assert.equal(fs.readFileSync(result.path, 'utf8'), 'fresh branch installer\n')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript refreshes an immutable-pin cache on a fresh install', async () => {
  const home = mkTmpHome()

  try {
    const commit = 'a'.repeat(40)
    const cached = cachedScriptPath(home, commit)
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'stale installer\n')

    const refs = []

    const result = await resolveInstallScript({
      installStamp: { commit, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      _download: async (ref, destPath) => {
        refs.push(ref)
        fs.writeFileSync(destPath, 'fresh pinned installer\n')

        return destPath
      }
    })

    assert.deepEqual(refs, [commit])
    assert.equal(result.source, 'download')
    assert.equal(result.commit, commit)
    assert.notEqual(result.path, cached)
    assert.equal(fs.readFileSync(cached, 'utf8'), 'stale installer\n')
    assert.equal(fs.readFileSync(result.path, 'utf8'), 'fresh pinned installer\n')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('each resolver owns immutable installer bytes across overlapping refs and retries', async () => {
  const home = mkTmpHome()

  try {
    const resolved = []

    for (const branch of ['feature/a', 'feature_a', 'feature/a']) {
      const script = await resolveInstallScript({
        installStamp: { commit: ZERO_COMMIT, branch },
        sourceRepoRoot: null,
        hermesHome: home,
        emit: () => {},
        _download: async (ref, destPath) => {
          fs.mkdirSync(path.dirname(destPath), { recursive: true })
          fs.writeFileSync(destPath, `installer for ${ref}: run ${resolved.length}`)

          return destPath
        }
      })
      resolved.push(script)
    }

    assert.equal(new Set(resolved.map(script => script.path)).size, 3, 'live runs cannot share an execution path')
    assert.equal(fs.readFileSync(resolved[0].path, 'utf8'), 'installer for feature/a: run 0')
    assert.equal(fs.readFileSync(resolved[1].path, 'utf8'), 'installer for feature_a: run 1')
    assert.equal(fs.readFileSync(resolved[2].path, 'utf8'), 'installer for feature/a: run 2')

    const before = fs.readdirSync(path.dirname(resolved[0].path)).sort()
    await assert.rejects(resolveInstallScript({
      installStamp: { commit: ZERO_COMMIT, branch: 'feature/a' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      _download: async (_ref, destPath) => {
        fs.writeFileSync(destPath, 'failed attempt')
        throw new Error('transport failed')
      }
    }), /transport failed/)
    assert.deepEqual(fs.readdirSync(path.dirname(resolved[0].path)).sort(), before)
    assert.equal(fs.readFileSync(resolved[0].path, 'utf8'), 'installer for feature/a: run 0')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript fails closed instead of executing an installed stale script', async () => {
  const home = mkTmpHome()

  try {
    const commit = 'a'.repeat(40)
    const scriptsDir = path.join(home, 'hermes-agent', 'scripts')
    fs.mkdirSync(scriptsDir, { recursive: true })
    fs.writeFileSync(path.join(scriptsDir, SCRIPT_NAME), 'stale installed script\n')

    await assert.rejects(
      resolveInstallScript({
        installStamp: { commit, branch: 'main' },
        sourceRepoRoot: null,
        hermesHome: home,
        emit: () => {},
        _download: async () => {
          throw new Error('Failed to download install script: HTTP 404')
        }
      }),
      /HTTP 404/
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('scriptUrls uses the site only for the exact branch it publishes', () => {
  assert.deepEqual(scriptUrls('main'), [
    `https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/${SCRIPT_NAME}`,
    `https://hermes-agent.nousresearch.com/${SCRIPT_NAME}`
  ])

  for (const ref of ['feature/#stable', 'release/版本%ready+candidate', 'a'.repeat(40), 'abcdef1', 'release', 'feature/install-fix']) {
    assert.equal(new URL(scriptUrls(ref)[0]).hash, '', 'branch characters must remain in the path')
    assert.deepEqual(scriptUrls(ref), [
      `https://raw.githubusercontent.com/NousResearch/hermes-agent/${ref.split('/').map(encodeURIComponent).join('/')}/scripts/${SCRIPT_NAME}`
    ])
  }
})

// Public test-only TLS fixture, used by the loopback server.
const TEST_TLS_KEY = `-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCZqS4ThThljkcI
JAYFnb5AncJ4TJHFYzeiGmpDyq/nE+PQ4TWMMobUqSlb8Z4Zivv98u3TRXbmkBJ4
La7bn8ggDhg5f20pVj8nO0jCOyvQOL8n5TEkBt0WKvmvENwPubMjG99/wqxwKRui
RBtfWkno7KIjl2lA6sgr6HhQkcjAnNrdEvyR+IDxPJYTfAwaZQu0tiCa6HxHAQKv
b2NOJYRR0KjTshLB9GDpH+71l6dCbmh6aU6FmACQanLo9WX61iyqnjB4qIXm8OSN
pY3W/BGzcOPO0nbeSjGRqBhBHPGwoAg0oOPZ1/VDkaS7AOupXkrdeBsq2NUnPmzl
dYxgZpxVAgMBAAECggEAFSqQPcEelSKllzH7IF/rupPgm1iUxddWbP5tf9wWIeME
ARxcl2zIVNfeahtct1EFSCRj7TPG3pie6q4ERZ17YCsA3D64xzZpqZpJefPTo7GF
Z1XzUG6fmrOdxCcy4Pmn+uCWh09GGIcZFt+B078oqiyaYwOyzG3q192EYTjLqfha
q1oqiG2kKkcG+cv68VfGKVHEbrNrYU9g/kjJuOswhknpLh+eMV8f7826MEgP+DQ2
JjhWLvpTuRBfmLHmko3B4jgKFfuZLa0sNEXpNu22yHwMswLYnjImlvCl3/MHwrCJ
y/asyye6smsoRuMkpHh4eGZ7NqaWT03DZ5qPuA0yywKBgQDHIwL1UEJUmUTgIjB6
Q5S6CvAWXgf07cIbR3sfhMXIkGDdgQwvGaaKS3PACXRTUDKtOFxhmEn/TIpyOTbx
M4Bs+nlxQH4+Zfc7zrbjFtV4czpfM9klkRazhhqLsKqFZ3MpFSBbbVHupsRwxiEJ
Sweynl9kEkoJU8dVrHXsyj0ZMwKBgQDFidvo9CXqGiRlOiIpFybEgnUgyHyX8hKd
WFXQv1LYrmHQkWx4Ty5A0WDdfiUp/1ucG+GcmDeC8swei5ofS51eQFhZeJMdDv7L
mV5/DkdNqFG1ORwfhKzw0huDccfb7Jth3drGOGpaUg33dDQMofCmbdThUSMjDsYB
Kf+LGEjEVwKBgFYMY/fa4X6q6B8txuLeFwM5PLt9kFSe9HRTM/nPpqNe9+xfGgO0
QsmZhv/hVfm2Ot+s7gZiBv+hdGWdIYeiaIkuxpFQe/y8lNOsJE0GjeHJcNy4i8l2
42dZuFjKUzToGdQTw/Kdz3yfZV0R0C6y1DWzx6Z3XLShFg6IQkC6tyIPAoGAJXQw
GAlCrxJp2C+fjn7vQM8jeiXJSd4CHYdELiI4iRD3Rt5r3JvWvz9zyEtErKPYMM8w
hcpurAtxHFGH1Ws22UoF9mDgM+BF+0CHJDwG1PiXFW9Qn8E+MSMFSHToWhCQnYu9
EVxc/ecU8tg7jjGeOVAVzurdaKZCcLIP28Ws9l0CgYB66bbvabqoU/hKco1Rer1x
t3fu2i+McL3D/XpCjnwFgWZO63uH6Zl5RuzDwnkeCAzbFwxHVorOSR2OQCm76Bjj
R/bd1EFeXn5ZL1D2JGbuYoqqU1JYB0F4Bbvp74A+90D6MUlLUBz+C2rNFSMFhr5T
jVn02OaufAlgk9Y7N4MWJg==
-----END PRIVATE KEY-----`
const TEST_TLS_CERT = `-----BEGIN CERTIFICATE-----
MIIDITCCAgmgAwIBAgIUTQdlMJH7EHLbIoMaGHystMrrhL8wDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MCAXDTI2MDkzMDE0NTgwNVoYDzIxMjYw
OTA2MTQ1ODA1WjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCZqS4ThThljkcIJAYFnb5AncJ4TJHFYzeiGmpDyq/n
E+PQ4TWMMobUqSlb8Z4Zivv98u3TRXbmkBJ4La7bn8ggDhg5f20pVj8nO0jCOyvQ
OL8n5TEkBt0WKvmvENwPubMjG99/wqxwKRuiRBtfWkno7KIjl2lA6sgr6HhQkcjA
nNrdEvyR+IDxPJYTfAwaZQu0tiCa6HxHAQKvb2NOJYRR0KjTshLB9GDpH+71l6dC
bmh6aU6FmACQanLo9WX61iyqnjB4qIXm8OSNpY3W/BGzcOPO0nbeSjGRqBhBHPGw
oAg0oOPZ1/VDkaS7AOupXkrdeBsq2NUnPmzldYxgZpxVAgMBAAGjaTBnMB0GA1Ud
DgQWBBT16ur+jx3XQDiqDpMD/ndiUQC8FTAfBgNVHSMEGDAWgBT16ur+jx3XQDiq
DpMD/ndiUQC8FTAPBgNVHRMBAf8EBTADAQH/MBQGA1UdEQQNMAuCCWxvY2FsaG9z
dDANBgkqhkiG9w0BAQsFAAOCAQEAiZVImwUs+R8tbdB6CSaRSwGPC4+ErKY/JsO3
CBcOCDR8DDdSFTSsBY1Xi1wyeuKJ3JvmNrQXW49jSvGAaHF1DrDLFK9k3VYK1xcy
cRDA7FEKzO9t1SFgwkH/zBiRpBoFNvHSqlxJdQCtjy4S/iFlDXxQNIGMDynOhTQf
o4/zI0tIlzAOWiNr7I3y1o1kK84bGtFR6ZVTl2s3fbY7z6oCa2nV6ONJs5yz4Fq7
G1a0DDvP/f5z/MkcMhomMul+U+2Figx9+aWV3VKOdPDLCAikMGJqG8yPNVoo6MdY
8PfEh1BQjPaMCOCmU1SFgUbQN1vTVw8ebHrJVcCQz06MEqvCzQ==
-----END CERTIFICATE-----`

// Route production HTTPS requests to real TLS. Redirects, socket failures,
// stream teardown and filesystem publication still follow Node's real path.
async function withInstallServer(
  handler: (req: IncomingMessage, res: ServerResponse) => void,
  run: (seen: string[], home: string) => Promise<void>
) {
  const home = mkTmpHome()
  const server = https.createServer({ key: TEST_TLS_KEY, cert: TEST_TLS_CERT }, handler)
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const { port } = server.address() as AddressInfo
  const seen: string[] = []
  const realGet = https.get.bind(https)
  const spy = vi.spyOn(https, 'get').mockImplementation((url, options, cb) => {
    const target = new URL(String(url))
    const requestOptions = typeof options === 'function' ? {} : options
    const onResponse = typeof options === 'function' ? options as (res: IncomingMessage) => void : cb
    seen.push(target.href)

    return realGet(new URL(`https://127.0.0.1:${port}${target.pathname}${target.search}`), {
      ...requestOptions,
      ca: TEST_TLS_CERT,
      servername: 'localhost',
      agent: false
    }, onResponse)
  })

  try {
    await run(seen, home)
  } finally {
    spy.mockRestore()
    server.closeAllConnections()
    await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()))
    fs.rmSync(home, { recursive: true, force: true })
  }
}

test('bootstrap keeps its run script through execution and removes only its own file on every outcome', async () => {
  let failing = false
  const manifest = '{"stages":[],"protocol_version":1}'
  const posix = `#!/bin/sh\nprintf '%s\n' '${manifest}'\n`
  const powershell = `param([switch]$Manifest, [string]$Commit, [string]$Branch)\nWrite-Output '${manifest}'\n`

  await withInstallServer((_req, res) => {
    res.end(failing ? 'exit 3\n' : process.platform === 'win32' ? powershell : posix)
  }, async (_seen, home) => {
    const commit = 'a'.repeat(40)
    const oldCache = cachedScriptPath(home, commit)
    fs.mkdirSync(path.dirname(oldCache), { recursive: true })
    fs.writeFileSync(oldCache, 'another live run')

    for (const outcome of ['complete', 'manifest-failure', 'cancel']) {
      failing = outcome === 'manifest-failure'
      const controller = new AbortController()
      let saved: string | undefined
      let presentAtComplete = false
      const result = await runBootstrap({
        installStamp: { commit, branch: 'main' },
        activeRoot: path.join(home, 'checkout'),
        sourceRepoRoot: null,
        hermesHome: home,
        logRoot: path.join(home, 'logs'),
        abortSignal: controller.signal,
        onEvent: ev => {
          if (ev.line?.startsWith('[bootstrap] saved to ')) {
            saved = ev.line.slice('[bootstrap] saved to '.length)
            assert.notEqual(saved, oldCache)
            assert.ok(fs.existsSync(saved))

            if (outcome === 'cancel') {
              controller.abort()
            }
          }

          if (ev.type === 'complete') {
            presentAtComplete = fs.existsSync(saved)
          }
        }
      })
      assert.ok(saved)
      assert.equal(result.ok, outcome === 'complete')
      assert.equal(presentAtComplete, outcome === 'complete', 'run file remains available until execution completes')
      assert.equal(fs.existsSync(saved), false, outcome)
      assert.equal(fs.readFileSync(oldCache, 'utf8'), 'another live run', outcome)
      assert.deepEqual(fs.readdirSync(path.dirname(oldCache)), [path.basename(oldCache)])
    }
  })
})

test('installer ladder preserves identity, old files and successful fallback attribution', async () => {
  let status = 403
  let fallbackStatus = 200
  const installer = '#!/bin/sh\necho site fallback\n'

  await withInstallServer((req, res) => {
    const isRaw = req.url?.startsWith('/NousResearch/')
    res.writeHead(isRaw ? status : fallbackStatus)
    res.end(isRaw ? 'edge unavailable' : installer)
  }, async (seen, home) => {
    const dest = path.join(home, 'download', SCRIPT_NAME)
    const events: { line?: string }[] = []
    const result = await resolveInstallScript({
      installStamp: { commit: ZERO_COMMIT, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: ev => events.push(ev)
    })
    assert.equal(fs.readFileSync(result.path, 'utf8'), DOWNLOAD_BOM + installer)
    assert.ok(events.some(ev => ev.line?.includes('from fallback https://hermes-agent.nousresearch.com/')))
    assert.equal(seen.length, 2)

    for (const retryable of [403, 429, 500, 503]) {
      status = retryable
      seen.length = 0
      assert.equal(await downloadInstallScript('main', dest), dest)
      assert.equal(seen.length, 2)
      assert.equal(fs.readFileSync(dest, 'utf8'), DOWNLOAD_BOM + installer)
    }

    for (const ref of ['feature/#stable', 'release/版本%ready+candidate', 'a'.repeat(40), 'abcdef1', 'release', 'feature/install-fix']) {
      status = 503
      seen.length = 0
      fs.writeFileSync(dest, 'previous successful installer')
      await assert.rejects(downloadInstallScript(ref, dest), /HTTP 503/)
      assert.equal(seen.length, 1)
      assert.equal(fs.readFileSync(dest, 'utf8'), 'previous successful installer')
    }

    for (const definitive of [400, 401, 404, 410]) {
      status = definitive
      seen.length = 0
      await assert.rejects(downloadInstallScript('main', dest), new RegExp(`HTTP ${definitive}`))
      assert.equal(seen.length, 1)
      assert.equal(fs.readFileSync(dest, 'utf8'), 'previous successful installer')
    }

    status = 403
    fallbackStatus = 503
    seen.length = 0
    const failedEvents: { line?: string }[] = []
    await assert.rejects(resolveInstallScript({
      installStamp: { commit: ZERO_COMMIT, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: ev => failedEvents.push(ev)
    }), /HTTP 403[\s\S]*HTTP 503/)
    assert.equal(seen.length, 2)
    assert.ok(!failedEvents.some(ev => ev.line?.includes('from fallback')))
    assert.equal(fs.readFileSync(result.path, 'utf8'), DOWNLOAD_BOM + installer)
    assert.deepEqual(fs.readdirSync(path.dirname(dest)), [SCRIPT_NAME])
  })
})

test('installer bodies reject corrupt text and error documents while preserving exact valid bytes', async () => {
  const limit = 2 * 1024 * 1024
  const complete = Buffer.from('#!/bin/sh\necho café 中文\n')
  let body = complete
  let contentType = 'text/plain'
  let advertisedLength: number | undefined
  let bothFail = false

  await withInstallServer((req, res) => {
    const isRaw = req.url?.startsWith('/NousResearch/')
    const selected = isRaw || bothFail ? body : complete
    const headers: Record<string, string | number> = { 'content-type': isRaw || bothFail ? contentType : 'text/plain' }

    if (isRaw && advertisedLength !== undefined) {
      headers['content-length'] = advertisedLength
    }

    res.writeHead(200, headers)
    // Split a UTF-8 code point: checks must span transport chunks.
    res.write(selected.subarray(0, 19))
    res.end(selected.subarray(19))
  }, async (seen, home) => {
    const dest = path.join(home, SCRIPT_NAME)
    const invalidBodies = [
      { bytes: Buffer.alloc(0), mime: 'text/plain' },
      { bytes: Buffer.from('\uFEFF \n\t'), mime: 'text/plain' },
      { bytes: Buffer.from('echo ok\0'), mime: 'text/plain' },
      { bytes: Buffer.from([0xc3, 0x28]), mime: 'text/plain' },
      { bytes: Buffer.from(' <HTML>proxy unavailable</HTML>'), mime: 'text/plain' },
      { bytes: Buffer.from('\uFEFF\n<!DOCTYPE HTML>proxy unavailable'), mime: 'text/plain' },
      { bytes: complete, mime: 'text/html; charset=utf-8' },
      { bytes: Buffer.from('{"error":"unavailable"}'), mime: 'application/json' },
      { bytes: Buffer.alloc(limit + 1, 'x'), mime: 'text/plain' }
    ]

    for (const invalid of invalidBodies) {
      body = invalid.bytes
      contentType = invalid.mime
      seen.length = 0
      fs.writeFileSync(dest, 'old installer')
      await downloadInstallScript('main', dest)
      assert.equal(seen.length, 2)
      assert.equal(fs.readFileSync(dest, 'utf8'), DOWNLOAD_BOM + complete.toString('utf8'))
      assert.deepEqual(fs.readdirSync(home), [SCRIPT_NAME])

      seen.length = 0
      fs.writeFileSync(dest, 'old installer')
      await assert.rejects(downloadInstallScript('a'.repeat(40), dest), /installer|UTF-8|document|NUL|bytes/i)
      assert.equal(seen.length, 1)
      assert.equal(fs.readFileSync(dest, 'utf8'), 'old installer')
      assert.deepEqual(fs.readdirSync(home), [SCRIPT_NAME])
    }

    body = Buffer.from('short body')
    contentType = 'text/plain'
    advertisedLength = limit + 1
    seen.length = 0
    await downloadInstallScript('main', dest)
    assert.equal(seen.length, 2, 'oversized Content-Length is rejected before waiting for the body')
    assert.equal(fs.readFileSync(dest, 'utf8'), DOWNLOAD_BOM + complete.toString('utf8'))
    advertisedLength = undefined

    bothFail = true
    body = Buffer.from('<html>no installer</html>')
    fs.writeFileSync(dest, 'old installer')
    await assert.rejects(downloadInstallScript('main', dest), /error document/)
    assert.equal(fs.readFileSync(dest, 'utf8'), 'old installer')
    assert.deepEqual(fs.readdirSync(home), [SCRIPT_NAME])
    bothFail = false

    for (const valid of [complete, Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), complete]), Buffer.alloc(limit, 'x')]) {
      body = valid
      seen.length = 0
      await downloadInstallScript('feature/#stable', dest)
      assert.equal(seen.length, 1)
      assert.equal(new URL(seen[0]).hash, '')
      assert.ok(new URL(seen[0]).pathname.includes('feature/%23stable/scripts/'))
      const expected = process.platform === 'win32' && !valid.subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf]))
        ? Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), valid])
        : valid
      assert.deepEqual(fs.readFileSync(dest), expected)
    }
  })
})

test('cached PowerShell uses one UTF-8 BOM and POSIX retains its shebang bytes', () => {
  const bom = Buffer.from([0xef, 0xbb, 0xbf])
  const powershell = Buffer.from('Write-Host "café 中文"\n')
  const shell = Buffer.from('#!/bin/sh\necho café\n')
  assert.deepEqual(prepareCachedScriptBytes('powershell', powershell), Buffer.concat([bom, powershell]))
  assert.deepEqual(prepareCachedScriptBytes('powershell', Buffer.concat([bom, powershell])), Buffer.concat([bom, powershell]))
  assert.deepEqual(prepareCachedScriptBytes('posix', shell), shell)
})

test('installer transfer bounds redirects and stalled bodies and never publishes partial or failed local writes', async () => {
  let mode = 'held-body'
  let relativeStatus = 307
  let finishBody: (() => void) | undefined
  const timers = new Set<ReturnType<typeof setInterval>>()
  const installer = '#!/bin/sh\necho complete\n'
  let cancelTransfer: (() => void) | undefined

  try {
    await withInstallServer((req, res) => {
      if (!req.url?.startsWith('/NousResearch/') && req.url !== '/relative-target') {
        res.end(installer)

        return
      }

      const handlers: Record<string, () => void> = {
        'held-body': () => {
          res.write(installer.slice(0, 4))
          finishBody = () => res.end(installer.slice(4))
        },
        reset: () => {
          req.socket.destroy()
        },
        relative: () => {
          if (req.url === '/relative-target') {
            res.end(installer)
          } else {
            res.writeHead(relativeStatus, { location: '/relative-target' })
            res.end()
          }
        },
        loop: () => {
          res.writeHead(302, { location: req.url })
          res.end()
        },
        'missing-location': () => {
          res.writeHead(302)
          res.end()
        },
        'http-redirect': () => {
          res.writeHead(301, { location: 'http://localhost/install.sh' })
          res.end()
        },
        truncated: () => {
          res.writeHead(200, { 'content-length': Buffer.byteLength(installer) + 100 })
          res.end(installer.slice(0, 4))
        },
        // The deadline includes waiting for headers.
        'stall-headers': () => {},
        'stall-body': () => {
          res.writeHead(200)
          res.write('#!')
        },
        trickle: () => {
          res.writeHead(200)
          const timer = setInterval(() => res.write('x'), 10)
          timers.add(timer)
          res.once('close', () => {
            clearInterval(timer)
            timers.delete(timer)
          })
        },
        cancel: () => {
          res.write('#!')
          cancelTransfer?.()
        }
      }

      const handle = handlers[mode] || (() => res.end(installer))
      handle()
    }, async (seen, home) => {
      const dest = path.join(home, 'download', SCRIPT_NAME)
      fs.mkdirSync(path.dirname(dest))
      fs.writeFileSync(dest, 'old installer')
      const opts = { timeoutMs: 2500, idleTimeoutMs: 2000 }
      const realCreateWriteStream = fs.createWriteStream.bind(fs)
      let writerOpened: () => void
      const openPromise = new Promise<void>(resolve => { writerOpened = resolve })
      const openSpy = vi.spyOn(fs, 'createWriteStream').mockImplementation((file, options) => {
        const out = realCreateWriteStream(file, options)
        out.once('open', () => writerOpened())

        return out
      })
      const heldDownload = downloadInstallScript('main', dest)

      try {
        await openPromise
        assert.equal(fs.readFileSync(dest, 'utf8'), 'old installer', 'active downloads cannot truncate the executable cache')
      } finally {
        finishBody?.()
        await heldDownload
        openSpy.mockRestore()
      }

      assert.equal(fs.readFileSync(dest, 'utf8'), DOWNLOAD_BOM + installer)
      mode = 'relative'

      for (const redirectStatus of [301, 302, 303, 307, 308]) {
        relativeStatus = redirectStatus
        seen.length = 0
        await downloadInstallScript('main', dest)
        assert.equal(seen.length, 2)
        assert.equal(fs.readFileSync(dest, 'utf8'), DOWNLOAD_BOM + installer)
      }

      for (const failure of ['reset', 'loop', 'missing-location', 'http-redirect', 'truncated', 'stall-headers', 'stall-body', 'trickle']) {
        mode = failure
        seen.length = 0
        assert.equal(await downloadInstallScript('main', dest, opts), dest, failure)
        assert.equal(fs.readFileSync(dest, 'utf8'), DOWNLOAD_BOM + installer, failure)
        assert.equal(seen.at(-1), `https://hermes-agent.nousresearch.com/${SCRIPT_NAME}`, failure)
        if (failure === 'loop') {
          assert.equal(seen.length, 7, 'five redirect hops then fallback')
        }
        assert.deepEqual(fs.readdirSync(path.dirname(dest)), [SCRIPT_NAME], failure)
      }

      mode = 'trickle'
      seen.length = 0
      fs.writeFileSync(dest, 'old installer')
      await assert.rejects(downloadInstallScript('a'.repeat(40), dest, opts), /deadline|abort|save/i)
      assert.equal(seen.length, 1)
      assert.equal(fs.readFileSync(dest, 'utf8'), 'old installer')
      assert.deepEqual(fs.readdirSync(path.dirname(dest)), [SCRIPT_NAME])

      mode = 'cancel'
      seen.length = 0
      const controller = new AbortController()
      cancelTransfer = () => controller.abort()
      await assert.rejects(resolveInstallScript({
        installStamp: { commit: ZERO_COMMIT, branch: 'main' },
        sourceRepoRoot: null,
        hermesHome: home,
        emit: () => {},
        abortSignal: controller.signal
      }), /deadline|abort|save/i)
      assert.equal(seen.length, 1, 'user cancellation stops before fallback')
      assert.equal(fs.readFileSync(dest, 'utf8'), 'old installer')
      assert.deepEqual(fs.readdirSync(path.dirname(dest)), [SCRIPT_NAME])

      mode = 'success'
      seen.length = 0
      const blockedParent = path.join(home, 'file-parent')
      fs.writeFileSync(blockedParent, 'keep')
      await assert.rejects(downloadInstallScript('main', path.join(blockedParent, SCRIPT_NAME)), /prepare/)
      assert.equal(seen.length, 0)
      assert.equal(fs.readFileSync(blockedParent, 'utf8'), 'keep')
      const realWriteStream = fs.createWriteStream.bind(fs)
      const writeSpy = vi.spyOn(fs, 'createWriteStream').mockImplementation((file, options) =>
        realWriteStream(path.dirname(String(file)), options))

      try {
        await assert.rejects(downloadInstallScript('main', dest), /Failed to save/)
        assert.equal(seen.length, 1, 'a real filesystem open failure never tries the site')
        assert.equal(fs.readFileSync(dest, 'utf8'), 'old installer')
      } finally {
        writeSpy.mockRestore()
      }

      seen.length = 0
      const directoryDest = path.join(home, 'download', 'existing-directory')
      fs.mkdirSync(directoryDest)
      await assert.rejects(downloadInstallScript('main', directoryDest), /publish/)
      assert.equal(seen.length, 1)
      assert.deepEqual(fs.readdirSync(path.dirname(dest)).sort(), [SCRIPT_NAME, 'existing-directory'].sort())
    })
  } finally {
    for (const timer of timers) {
      clearInterval(timer)
    }
  }
}, 30_000)

// #112675: install.sh colours its banners and curl/uv redraw progress with \r
// even into a pipe; the overlay renders lines as plain text, so the emitter
// must hand every consumer (log ring, Details panel, Copy output) the text a
// terminal would be left showing.
test('installer log lines reach the emitter without escape sequences; \\r redraws keep the last frame', () => {
  assert.equal(cleanInstallerLogLine('\u001b[0;32m✓\u001b[0m Detected: macos (macos)'), '✓ Detected: macos (macos)')
  assert.equal(cleanInstallerLogLine('\u001b[2K\u001b[1GCloning repository…\u001b[K'), 'Cloning repository…')
  assert.equal(cleanInstallerLogLine('\u001b]0;hermes\u0007Installing Hermes'), 'Installing Hermes')
  assert.equal(cleanInstallerLogLine('\r 12%\r 67%\r100%\u001b[K'), '100%')
  assert.equal(cleanInstallerLogLine('Resolving dependencies…\r'), 'Resolving dependencies…')
  // Only-escape frames drop entirely, so the caller emits nothing for them.
  assert.equal(cleanInstallerLogLine('\u001b[0m\r'), '')
  // Plain multi-byte text is untouched.
  assert.equal(cleanInstallerLogLine('Ready — café ✓ 中文'), 'Ready — café ✓ 中文')
})

test.skipIf(process.platform === 'win32')(
  'a manifest-step failure surfaces the installer tail without escape sequences',
  async () => {
    const home = mkTmpHome()
    fs.mkdirSync(path.join(home, 'scripts'))
    fs.writeFileSync(
      path.join(home, 'scripts', 'install.sh'),
      '#!/usr/bin/env bash\nprintf "\\033[0;31m\\xe2\\x9c\\x97\\033[0m manifest broke\\n" >&2\nexit 3\n'
    )

    const result = await runBootstrap({
      installStamp: null,
      activeRoot: home,
      sourceRepoRoot: home,
      hermesHome: home,
      logRoot: home,
      onEvent: () => {}
    })

    assert.equal(result.ok, false)
    assert.equal(result.error, 'install.sh --manifest failed: exit 3\n✗ manifest broke')
  }
)
