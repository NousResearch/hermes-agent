import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from './mock-server'
import { expect, type Page, test } from './test'

// FILE PREVIEW end-to-end: selecting files in the workspace tree opens them
// INSIDE the app's preview rail; Markdown renders formatted by default; the
// spot editor writes through the REAL Electron bridge (disk contents are
// asserted from this process, not from any renderer state).

let fixture: MockBackendFixture | null = null

function createWorkspace(root: string): string {
  const repo = path.join(root, 'workspace')

  fs.mkdirSync(repo, { recursive: true })
  execFileSync('git', ['init', '--initial-branch=main'], { cwd: repo })
  execFileSync('git', ['config', 'user.email', 'e2e@example.com'], { cwd: repo })
  execFileSync('git', ['config', 'user.name', 'Hermes E2E'], { cwd: repo })

  fs.writeFileSync(
    path.join(repo, 'README.md'),
    ['# E2E Preview', '', '- alpha', '- beta', '', '```sh', 'echo hello', '```'].join('\n'),
    'utf8'
  )
  fs.writeFileSync(path.join(repo, 'notes.txt'), 'plain notes body', 'utf8')
  fs.mkdirSync(path.join(repo, 'src'), { recursive: true })
  fs.writeFileSync(path.join(repo, 'src', 'util.ts'), 'export const x = 1\n', 'utf8')

  execFileSync('git', ['add', '.'], { cwd: repo })
  execFileSync('git', ['commit', '-m', 'initial'], { cwd: repo })

  return repo
}

test.beforeAll(async () => {
  const sandbox = createSandbox('file-preview')
  const repo = createWorkspace(sandbox.root)
  const mock = await startMockServer()

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nterminal:\n  cwd: ${repo}\n`, 'utf8')
  writeEnvFile(sandbox.hermesHome)

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  fixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }

  await waitForAppReady(fixture, 120_000)

  // The session resolves its cwd on the first turn; the Files pane shows that
  // cwd's tree. One throwaway turn against the mock model is enough.
  const composer = fixture.page.locator('[contenteditable="true"]').first()
  await composer.click()
  await composer.type('open the workspace', { delay: 2 })
  await fixture.page.keyboard.press('Enter')
  await fixture.page.waitForFunction(
    prompt => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(prompt),
    'open the workspace',
    { timeout: 15_000 }
  )
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

async function openFilesPane(): Promise<void> {
  const page = fixture!.page
  const tree = page.locator('[data-project-tree]')

  if ((await tree.count()) === 0) {
    // mod+j = Cmd+J on macOS / Ctrl+J elsewhere ('view.toggleRightSidebar').
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+J' : 'Control+J')
    await expect(tree).toBeVisible({ timeout: 10_000 })
  }
}

async function clickFile(name: string): Promise<void> {
  const page = fixture!.page

  await page.locator(`[data-project-tree] [title="${path.join(fixture!.sandbox.root, 'workspace', name)}"]`).click()
}

test.describe('file preview e2e', () => {
  // The mode-switcher/edit controls render uppercase labels ('SOURCE', 'EDIT').
  // Anchored case-insensitive regexes match those exactly while never
  // substring-matching chrome like "Layout editor" or "Edit message". Every
  // open preview TAB stays mounted, so multiple EDIT/SAVE/CANCEL controls
  // exist at once; the just-fronted tab is the most recently appended one
  // (openPreview appends), so .last() addresses ITS control.
  const previewButton = (page: Page, name: string) =>
    page.getByRole('button', { name: new RegExp(`^${name}$`, 'i') }).last()

  test('single-click opens a markdown file formatted inside the app', async () => {
    const page = fixture!.page

    await openFilesPane()
    await clickFile('README.md')

    // Formatted, not raw: an H1 exists and the fence became a code block.
    await expect(page.getByRole('heading', { level: 1, name: 'E2E Preview' })).toBeVisible({ timeout: 15_000 })
  })

  test('source mode shows raw markdown', async () => {
    const page = fixture!.page

    await openFilesPane()
    await clickFile('README.md')
    await previewButton(page, 'source').click()
    await expect(page.locator('.preview-source-code')).toContainText('# E2E Preview')
  })

  test('edit + save persists the draft to real disk', async () => {
    const page = fixture!.page
    const readmePath = path.join(fixture!.sandbox.root, 'workspace', 'README.md')

    await openFilesPane()
    await clickFile('README.md')
    await previewButton(page, 'edit').click()

    const editor = page.locator('.cm-content')
    await editor.click()
    await page.keyboard.press('ControlOrMeta+a')
    await page.keyboard.type('# Edited by E2E')

    await previewButton(page, 'save').click()

    await expect.poll(() => fs.readFileSync(readmePath, 'utf8'), { timeout: 10_000 }).toContain('# Edited by E2E')
  })

  test('cancel discards the draft and leaves disk untouched', async () => {
    const page = fixture!.page
    const notesPath = path.join(fixture!.sandbox.root, 'workspace', 'notes.txt')
    const before = fs.readFileSync(notesPath, 'utf8')

    // notes.txt was never written by earlier tests, so no watcher/save churn
    // can remount the editor under us mid-flow.
    await openFilesPane()
    await clickFile('notes.txt')
    await previewButton(page, 'edit').click()

    // Only the FRONTED tab's editor is interactable; background preview tabs
    // keep their (hidden) CodeMirror surfaces mounted too.
    const editor = page.locator('.cm-content').filter({ visible: true })
    await expect(editor).toHaveCount(1)
    await expect(editor).toBeVisible()
    await editor.click()
    await page.keyboard.press('ControlOrMeta+a')
    await page.keyboard.type('# SHOULD NOT PERSIST')

    await previewButton(page, 'cancel').click()

    expect(fs.readFileSync(notesPath, 'utf8')).toBe(before)
  })

  // Preview tabs cache their loaded snapshot: re-selecting a file re-FRONTS
  // its existing tab rather than re-reading disk (openPreview's documented
  // identity rule). Pin that switching between two files gives each its own
  // tab and switching back restores the first view.
  test('a second file opens as its own tab and switching back re-fronts the first', async () => {
    const page = fixture!.page

    await openFilesPane()

    await clickFile('notes.txt')
    await expect(page.locator('body')).toContainText('plain notes body')

    await clickFile('README.md')
    // Test 3's save self-reloaded THIS tab, so it shows the saved content
    // ('# Edited by E2E' — a paragraph, hence no H1). Switching back must
    // restore that view rather than showing notes.txt.
    await expect(page.locator('body')).toContainText('Edited by E2E')

    await clickFile('notes.txt')
    await expect(page.locator('body')).toContainText('plain notes body')
  })
})
