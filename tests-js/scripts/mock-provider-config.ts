import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import yaml from 'js-yaml'
import { z } from 'zod'

const section = z.object({}).passthrough()
const configSchema = z.object({
  model: section.optional(),
  providers: section.optional(),
  auxiliary: z.object({ title_generation: section.optional() }).passthrough().optional(),
  approvals: section.optional(),
  display: section.optional(),
}).passthrough()

export function validateMockUrl(value: string): string {
  const url = new URL(value)
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port
      || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('Mock URL must be a credential-free http://127.0.0.1:PORT origin')
  }
  return url.origin
}

/** Merge only test-owned settings. Do not replace the update feed or plugins. */
export function writeMockProviderConfig(
  hermesHome: string,
  mockUrl: string,
  extraDisplayConfig?: string,
  extraConfig?: string,
  modelContextLength?: number,
): void {
  const url = validateMockUrl(mockUrl)
  fs.mkdirSync(hermesHome, { recursive: true })
  const configPath = path.join(hermesHome, 'config.yaml')
  const config = configSchema.parse(fs.existsSync(configPath) ? yaml.load(fs.readFileSync(configPath, 'utf8')) ?? {} : {})
  const extra = configSchema.parse(extraConfig ? yaml.load(extraConfig) ?? {} : {})
  const display = section.parse(extraDisplayConfig ? yaml.load(extraDisplayConfig) ?? {} : {})
  const merged = {
    ...config,
    model: { ...config.model, default: 'mock-model', provider: 'mock', context_length: modelContextLength ?? 64000 },
    providers: {
      ...config.providers,
      mock: { api: `${url}/v1`, name: 'Mock', api_mode: 'chat_completions', key_env: 'MOCK_API_KEY', models: { 'mock-model': {} }, context_length: 64000 },
    },
    auxiliary: { ...config.auxiliary, title_generation: { ...config.auxiliary?.title_generation, enabled: false } },
    approvals: { ...config.approvals, mode: 'off' },
    ...extra,
  }
  if (extraDisplayConfig) {
    merged.display = { ...config.display, ...extra.display, ...display }
  }
  fs.writeFileSync(configPath, yaml.dump(merged), 'utf8')
}

/** Keep journey-owned entries, replacing only the inert test key. */
export function writeEnvFile(hermesHome: string, apiKey = 'e2e-mock-key'): void {
  if (!/^[\w-]+$/.test(apiKey)) {
    throw new Error('Mock key must be an inert single-line test value')
  }
  const envPath = path.join(hermesHome, '.env')
  const prior = fs.existsSync(envPath) ? fs.readFileSync(envPath, 'utf8') : ''
  const lines = prior.split(/\r?\n/).filter((line: string): boolean => !/^\s*(?:export\s+)?MOCK_API_KEY\s*=/.test(line))
  fs.writeFileSync(envPath, `${lines.join('\n').trimEnd()}\nMOCK_API_KEY=${apiKey}\n`, { mode: 0o600 })
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [home, url] = process.argv.slice(2)
  if (!home || !url || !path.isAbsolute(home)) {
    throw new Error('usage: node mock-provider-config.ts ABSOLUTE_HERMES_HOME MOCK_URL')
  }
  writeMockProviderConfig(home, url)
  writeEnvFile(home)
}