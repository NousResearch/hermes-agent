import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const desktopRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const packageJsonPath = resolve(desktopRoot, 'package.json')

async function readPackageJson() {
  return JSON.parse(await readFile(packageJsonPath, 'utf8'))
}

describe('desktop package build metadata', () => {
  it('keeps the real desktop app visible in the macOS Dock', async () => {
    const packageJson = await readPackageJson()

    expect(packageJson.build.appId).toBe('com.nousresearch.hermes')
    expect(packageJson.build.mac.extendInfo).not.toHaveProperty('LSUIElement')
  })
})