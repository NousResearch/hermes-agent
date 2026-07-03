import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { ESLint } from 'eslint'
import { describe, expect, it } from 'vitest'

import config from './eslint.config.mjs'

const dirname = path.dirname(fileURLToPath(import.meta.url))

function eslintForDesktop() {
  return new ESLint({
    cwd: dirname,
    ignore: false,
    overrideConfig: config,
    overrideConfigFile: true
  })
}

async function lintFixture(source, filePath) {
  const [result] = await eslintForDesktop().lintText(source, { filePath: path.join(dirname, filePath) })

  return result.messages.map(message => message.message)
}

describe('desktop eslint remote-file boundaries', () => {
  it('fires on BUG-E-shaped local filesystem escape hatches even with inline disables', async () => {
    const messages = await lintFixture(
      `/* eslint-disable custom-rules/no-local-desktop-file-ops */
import { revealDesktopPath } from '@/lib/desktop-fs'

export async function bad() {
  await revealDesktopPath('/gateway/repo')
  await window.hermesDesktop.readFileDataUrl('/gateway/repo/image.png')
}
`,
      'src/app/chat/sidebar/projects/bad-boundary.tsx'
    )

    expect(messages).toContain(
      "Do not import local-only desktop-fs export 'revealDesktopPath' here. Route through a remote-aware facade or guard remote mode with an honest message."
    )
    expect(messages).toContain(
      'Do not call window.hermesDesktop.readFileDataUrl outside an allowlisted local-file seam; use readDesktopFileDataUrl or another remote-aware facade.'
    )
  })

  it('spares remote-aware desktop filesystem facades', async () => {
    const messages = await lintFixture(
      `import { readDesktopFileDataUrl, revealDesktopFile } from '@/lib/desktop-fs'

export async function good() {
  await revealDesktopFile('/gateway/repo')

  return readDesktopFileDataUrl('/gateway/repo/image.png')
}
`,
      'src/app/chat/sidebar/projects/good-boundary.tsx'
    )

    expect(messages).toEqual([])
  })
})
