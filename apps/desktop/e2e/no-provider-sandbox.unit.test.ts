import * as http from 'node:http'

import { afterEach, describe, expect, test } from 'vitest'
import { parse } from 'yaml'

import { type EgressRecorder, startEgressRecorder } from './egress-recorder'
import {
  chromiumProxyArgs,
  NO_PROVIDER_CONFIG,
  noProviderNetworkEnv,
  nousAccountAttempts,
  offlineViolations
} from './no-provider-sandbox'

// A sandbox without a provider is a fresh desktop install: unless guests are off, its first-run
// onboarding claims the welcome trial at the Nous portal (a real guest account, then a token and a
// browser check) from the developer's machine. These tests pin what keeps it offline.

const read = (yaml: string) => parse(yaml) as Record<string, Record<string, unknown> | undefined>

describe('no-provider sandbox config', () => {
  test('switches guests off, so no welcome trial or connector guest is ever minted', () => {
    expect(read(NO_PROVIDER_CONFIG).nous?.guest).toBe(false)
  })

  test('configures no provider and pins the UI to English', () => {
    const config = read(NO_PROVIDER_CONFIG)

    expect(config.model).toBeUndefined()
    expect(config.providers).toBeUndefined()
    expect(config.display?.language).toBe('en')
  })
})

describe('no-provider sandbox network env', () => {
  const recorder = 'http://127.0.0.1:43123'
  const args = chromiumProxyArgs(recorder)

  test('points every Nous portal and inference override and every proxy at loopback', () => {
    const env = noProviderNetworkEnv(recorder)

    expect(offlineViolations(NO_PROVIDER_CONFIG, env, args)).toEqual([])

    // The names the backend reads, spelled out here so that dropping one from the module fails.
    for (const key of ['HERMES_PORTAL_BASE_URL', 'NOUS_PORTAL_BASE_URL', 'NOUS_INFERENCE_BASE_URL']) {
      expect(new URL(env[key] ?? 'missing:').hostname, key).toBe('127.0.0.1')
    }

    for (const key of ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy']) {
      expect(env[key], key).toBe(recorder)
    }
  })

  test('flags a launch whose env would route the Nous portal or inference to a real host', () => {
    const env = {
      ...noProviderNetworkEnv(recorder),
      HERMES_PORTAL_BASE_URL: 'https://portal.nousresearch.com',
      NOUS_INFERENCE_BASE_URL: 'https://welcome-api.nousresearch.com/v1'
    }

    expect(offlineViolations(NO_PROVIDER_CONFIG, env, args)).toEqual([
      'HERMES_PORTAL_BASE_URL=https://portal.nousresearch.com is not loopback',
      'NOUS_INFERENCE_BASE_URL=https://welcome-api.nousresearch.com/v1 is not loopback'
    ])
  })

  test("flags a launch where the developer's own lowercase proxy would win", () => {
    const env = { ...noProviderNetworkEnv(recorder), https_proxy: 'http://proxy.example:3128' }

    expect(offlineViolations(NO_PROVIDER_CONFIG, env, args)).toEqual([
      'https_proxy=http://proxy.example:3128 is not loopback'
    ])
  })

  test('routes the app itself (renderer and main process) to the recorder too', () => {
    // Chromium on macOS and Windows follows the system proxy settings, not the variables above.
    expect(args).toEqual([`--proxy-server=${recorder}`])
  })

  test('flags a launch without the proxy switch, or with one that leaves the machine', () => {
    const env = noProviderNetworkEnv(recorder)

    expect(offlineViolations(NO_PROVIDER_CONFIG, env, ['--disable-gpu'])).toEqual([
      'the app is launched without --proxy-server=<recorder>'
    ])
    expect(offlineViolations(NO_PROVIDER_CONFIG, env, ['--proxy-server=http://proxy.example:3128'])).toEqual([
      '--proxy-server=http://proxy.example:3128 is not loopback'
    ])
  })

  test('flags a launch without the overrides or with guests on', () => {
    const { NOUS_PORTAL_BASE_URL: _dropped, ...env } = noProviderNetworkEnv(recorder)

    expect(offlineViolations('# no providers\n', env, args)).toEqual([
      'config.yaml does not set nous.guest: false',
      'NOUS_PORTAL_BASE_URL is unset'
    ])
  })
})

describe('nousAccountAttempts', () => {
  test('picks the attempts at the Nous portal and inference hosts out of everything a sandbox tried', () => {
    expect(
      nousAccountAttempts([
        'models.dev:443',
        'portal.nousresearch.com:443',
        'hermes-agent.nousresearch.com:443',
        'welcome-api.nousresearch.com:443',
        'INFERENCE-API.nousresearch.com:443'
      ])
    ).toEqual(['portal.nousresearch.com:443', 'welcome-api.nousresearch.com:443', 'INFERENCE-API.nousresearch.com:443'])
  })
})

describe('egress recorder', () => {
  let recorder: EgressRecorder | null = null

  afterEach(async () => {
    await recorder?.close()
    recorder = null
  })

  const tunnel = (proxy: URL, target: string) =>
    new Promise<number>((resolve, reject) => {
      http
        .request({ host: proxy.hostname, port: proxy.port, method: 'CONNECT', path: target })
        .on('connect', response => {
          response.destroy()
          resolve(response.statusCode ?? 0)
        })
        .on('error', reject)
        .end()
    })

  const plain = (proxy: URL, target: string) =>
    new Promise<number>((resolve, reject) => {
      http
        .get(
          { host: proxy.hostname, port: proxy.port, path: target, headers: { host: new URL(target).host } },
          response => {
            response.resume()
            resolve(response.statusCode ?? 0)
          }
        )
        .on('error', reject)
    })

  test('refuses every proxied request with 503 and records only its host', async () => {
    recorder = await startEgressRecorder()
    const proxy = new URL(recorder.url)

    expect(await tunnel(proxy, 'portal.nousresearch.com:443')).toBe(503)
    expect(await plain(proxy, 'http://welcome-api.nousresearch.com/v1/models')).toBe(503)
    expect(recorder.hosts()).toEqual(['portal.nousresearch.com:443', 'welcome-api.nousresearch.com:80'])
  })
})
