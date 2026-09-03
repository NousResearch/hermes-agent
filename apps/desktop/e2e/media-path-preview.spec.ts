import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const EXPECTED_HEADING = 'Monark I13+ Time Thesis Deep Research Plan'
const PROMPT = 'Return the configured MEDIA attachment.'

let fixture: MockBackendFixture | null = null
let generatedRoot: string | null = null
let targetPath = ''

test.beforeAll(async () => {
  const requested = process.env.HERMES_MEDIA_E2E_PATH?.trim()

  if (requested) {
    if (!path.isAbsolute(requested) || !fs.statSync(requested).isFile()) {
      throw new Error(`HERMES_MEDIA_E2E_PATH must name an existing absolute file: ${requested}`)
    }

    targetPath = requested
  } else {
    generatedRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'Hermes Projects-media-preview-'))
    targetPath = path.join(
      generatedRoot,
      'Monark-Inc',
      'Bank-Charter',
      '.hermes',
      'plans',
      '2026-09-03_204030-monark-i13-plus-time-thesis-deep-research.md'
    )
    fs.mkdirSync(path.dirname(targetPath), { recursive: true })
    fs.writeFileSync(targetPath, `# ${EXPECTED_HEADING}\n\nExact-path E2E fixture.\n`, 'utf8')
  }

  fixture = await setupMockBackend({ mockServer: { reply: `MEDIA:${targetPath}` } })
  await waitForAppReady(fixture)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null

  if (generatedRoot) {
    fs.rmSync(generatedRoot, { force: true, recursive: true })
    generatedRoot = null
  }
})

test('streams a spaced MEDIA path into the file card and opens the markdown preview', async () => {
  const page = fixture!.page
  const composer = page.locator('[contenteditable="true"]').first()

  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.fill(PROMPT)
  await page.keyboard.press('Enter')

  const filename = path.basename(targetPath)
  const fileCardName = page.getByText(filename, { exact: true }).first()

  await fileCardName.waitFor({ state: 'visible', timeout: 60_000 })
  await expect(page.getByRole('button', { name: 'Download' }).last()).toBeVisible()

  const openPreview = page.getByRole('button', { name: 'Open preview' }).last()

  await expect(openPreview).toBeVisible()
  await openPreview.click()

  const rightRail = page.getByRole('complementary').last()

  await expect(rightRail).toBeVisible()
  await expect(rightRail.getByText(EXPECTED_HEADING, { exact: true })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: 'Hide' }).last()).toBeVisible()
})
