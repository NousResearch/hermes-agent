#!/usr/bin/env node
import { appendFile } from 'node:fs/promises'
import { validateBundleInputs } from './bundle-inputs.mjs'
import { bundledMatrix } from '../../../scripts/sandbox/generate-e2e-matrix.mjs'

const route = process.env.BUNDLE_ROUTE || 'all'
for (const platform of ['windows', 'macos']) {
  const url = process.env[`BUNDLE_${platform.toUpperCase()}_MANIFEST`] || ''
  const selected = ['all', 'bundled', `${platform}-bundled`].includes(route)
  if (!url || !selected) {
    if (selected && ['bundled', `${platform}-bundled`].includes(route)) {
      throw new Error(`${platform}: a pinned bundle manifest is required for this route`)
    }
    await appendFile(process.env.GITHUB_OUTPUT, `${platform}={"include":[]}\n${platform}-arch=\n`)
    await appendFile(process.env.GITHUB_STEP_SUMMARY, `\n${platform} bundled update: not run (${!url ? 'no signed package pair supplied' : 'route not selected'}).\n`)
    continue
  }
  const location = new URL(url)
  if (location.protocol !== 'https:' || location.username || location.password) throw new Error('CI manifests require HTTPS without credentials')
  const response = await fetch(location, { signal: AbortSignal.timeout(60_000) })
  if (!response.ok) throw new Error(`Bundle manifest HTTP ${response.status}`)
  const text = await response.text()
  if (text.length > 1024 * 1024) throw new Error('Bundle manifest exceeds size limit')
  const data = JSON.parse(text)
  const manifest = validateBundleInputs(data, platform, data.arch)
  if (manifest.new.commit !== process.env.GITHUB_SHA) throw new Error(`${platform}: candidate commit must equal the tested workflow SHA`)
  await appendFile(process.env.GITHUB_OUTPUT, `${platform}=${JSON.stringify(bundledMatrix(platform, manifest.old.tag))}\n${platform}-arch=${manifest.arch}\n`)
  await appendFile(process.env.GITHUB_STEP_SUMMARY, `\n${platform}/${manifest.arch} packaged-app → open-app-update: ${manifest.old.tag} → ${manifest.new.tag} (${manifest.new.commit}).\n`)
}
