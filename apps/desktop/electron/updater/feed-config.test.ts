import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, it } from 'vitest'

import { readUpdatesFeedBaseFromConfig, resolveFeedBaseUrl } from './feed-config'

it('reads only the updates feed across equivalent YAML representations', (): void => {
  const home: string = mkdtempSync(join(tmpdir(), 'hermes-feed-config-'))
  const config: string = join(home, 'config.yaml')
  const expected: string = 'https://mirror.example/updates'

  const documents: string[] = [
    `updates:\n  desktop_feed_base_url: ${expected} # mirror\n`,
    `updates: {desktop_feed_base_url: '${expected}'}\n`,
    `other:\n  desktop_feed_base_url: https://other.example\nupdates:\n  desktop_feed_base_url: "${expected}"\n`,
    `updates:\n  desktop_feed_base_url: >-\n    ${expected}\n`,
    `\uFEFFupdates:\n  desktop_feed_base_url: ${expected}\n`
  ]

  try {
    for (const document of documents) {
      writeFileSync(config, document)
      expect(readUpdatesFeedBaseFromConfig(config)).toBe(expected)
    }
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})

it('leaves fallback selection available for missing, invalid or non-string config', (): void => {
  const home: string = mkdtempSync(join(tmpdir(), 'hermes-feed-config-'))
  const config: string = join(home, 'config.yaml')

  const documents: string[] = [
    '',
    'updates: [',
    'null',
    'updates: null',
    'updates: []',
    'other: {desktop_feed_base_url: https://other.example}',
    'updates: {desktop_feed_base_url: false}',
    'updates: {desktop_feed_base_url: 123}',
    'updates: {desktop_feed_base_url: [https://mirror.example]}'
  ]

  try {
    expect(readUpdatesFeedBaseFromConfig(config)).toBe('')

    for (const document of documents) {
      writeFileSync(config, document)
      expect(readUpdatesFeedBaseFromConfig(config)).toBe('')
    }
  } finally {
    rmSync(home, { recursive: true, force: true })
  }
})

it('holds a configured feed base to the update-origin rule and refuses the rest', (): void => {
  expect(resolveFeedBaseUrl('')).toBe('')
  expect(resolveFeedBaseUrl('  ')).toBe('')
  expect(resolveFeedBaseUrl('https://mirror.example/updates')).toBe('https://mirror.example/updates')
  expect(resolveFeedBaseUrl('http://127.0.0.1:8443/feed')).toBe('http://127.0.0.1:8443/feed')

  for (const bad of [
    'http://evil.example/feed',
    'https://user:pw@mirror.example/feed',
    'https://mirror.example/../feed'
  ]) {
    expect((): string => resolveFeedBaseUrl(bad)).toThrow()
  }
})
