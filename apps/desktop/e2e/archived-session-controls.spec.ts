/**
 * E2E (FRAMEWORK-07): an ARCHIVED cross-profile session hydrates its session
 * controls instead of failing closed with
 * `Session controls unavailable: Session owner could not be resolved …`.
 *
 * The reported shape: the window's active profile is `default`, the session
 * lives on `arthur`, and it is archived — so it is in no sidebar slice, no
 * resume hint and no tile route, which used to leave `session.control.read`
 * with no owner to route to. Opening it from the Archived view must reach the
 * OWNING profile's backend and paint the controls, with no owner-resolution
 * banner.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists, and
 * `HERMES_DESKTOP_PYTHON` must be an interpreter that can import the repo's
 * `hermes_state` (PYTHONPATH is set to the repo root for the seed).
 */

import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { expect, test } from './test'

const SESSION_ID = '20261105_090500_fw07arch'
const TITLE = 'FW07 archived cross-profile'

// Seeds an archived row on the SECOND profile through the real SessionDB — the
// same writer the backend uses — then stamps the two fields the report named:
// profile_name='arthur', archived=1 (auto_archived=1, end_reason=session_reset).
const SEED = `
import os, sys
from pathlib import Path
from hermes_state import SessionDB

home = Path(os.environ['HERMES_HOME'])
db_path = home / 'profiles' / 'arthur' / 'state.db'

with SessionDB(db_path=db_path) as db:
    db.create_session(sys.argv[1], source='telegram', model='e2e-mock', cwd=os.environ['HOME'])
    db.set_session_title(sys.argv[1], sys.argv[2])
    db.append_message(sys.argv[1], 'user', 'archived question')
    db.append_message(sys.argv[1], 'assistant', 'archived answer')
    db._conn.execute(
        'UPDATE sessions SET profile_name=?, archived=1, auto_archived=1, end_reason=? WHERE id=?',
        ('arthur', 'session_reset', sys.argv[1]),
    )
    db._conn.commit()
`

test('an archived cross-profile session hydrates its session controls (FRAMEWORK-07)', async () => {
  test.setTimeout(300_000)
  const sandbox = createSandbox('fw07-archived-controls')
  const mock = await startMockServer()
  const python = process.env.HERMES_DESKTOP_PYTHON!
  const root = path.resolve(import.meta.dirname, '../../..')
  const env = buildAppEnv(sandbox, { HERMES_DESKTOP_PYTHON: python, TIRITH_ENABLED: 'false' })
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  // The custom provider reads OPENAI_API_KEY: without it the runtime fails
  // credential resolution and the app boots into the provider-setup modal.
  writeEnvFile(sandbox.hermesHome, 'e2e-mock-key', mock.url)

  const arthurHome = path.join(sandbox.hermesHome, 'profiles', 'arthur')
  fs.mkdirSync(arthurHome, { recursive: true })
  writeMockProviderConfig(arthurHome, mock.url)
  writeEnvFile(arthurHome, 'e2e-mock-key', mock.url)

  const seed = spawnSync(python, ['-c', SEED, SESSION_ID, TITLE], {
    cwd: root,
    env: { ...env, PYTHONPATH: root },
    encoding: 'utf8'
  })

  expect(seed.status, seed.stderr).toBe(0)

  // An existing install has a startup profile; without one the first-launch
  // migration guesses from the newest state.db, which here is arthur's.
  fs.writeFileSync(path.join(sandbox.userDataDir, 'active-profile.json'), JSON.stringify({ profile: 'default' }))

  const evidence = process.env.HERMES_E2E_EVIDENCE || path.join(sandbox.root, 'evidence')
  fs.mkdirSync(evidence, { recursive: true })

  let app: Awaited<ReturnType<typeof launchDesktop>>['app'] | undefined

  try {
    const launched = await launchDesktop(env)
    app = launched.app
    const page = launched.page

    // Both profiles on the rail: the window is on `default`, the session is on `arthur`.
    await expect(page.getByRole('button', { name: /arthur/ }).first()).toBeVisible({ timeout: 90_000 })
    console.log('STEP: profile rail shows arthur')

    // No sidebar slice lists an archived row (archived=exclude), so open it
    // through the desktop's own session-window bridge — deliberately WITHOUT
    // the profile option: an explicit profile would hand the tile an owner
    // route and bypass exactly the resolution path this card is about.
    const opened = await page.evaluate(async id => {
      const desktop = (window as { hermesDesktop?: { openSessionWindow?: (id: string) => Promise<unknown> } })
        .hermesDesktop

      try {
        return await desktop?.openSessionWindow?.(id)
      } catch (error) {
        return 'ERR ' + String(error)
      }
    }, SESSION_ID)

    console.log('STEP: openSessionWindow -> ' + JSON.stringify(opened))

    // The session may open in a second window or retarget this one.
    let target = page
    const deadline = Date.now() + 30_000

    while (Date.now() < deadline) {
      const other = app?.windows().find(candidate => candidate !== page)

      if (other) {
        target = other

        break
      }

      const titledHere = await page
        .getByText(TITLE, { exact: true })
        .first()
        .isVisible()
        .catch(() => false)

      if (titledHere) {
        break
      }

      await new Promise(resolve => setTimeout(resolve, 500))
    }

    console.log('STEP: target window selected (second window: ' + (target !== page) + ')')

    // The transcript must open (the open path probes, so this is expected even
    // before the fix) — the CONTROLS are what used to fail.
    const transcript = target.locator('[data-slot="aui_thread-viewport"]')
    await expect(transcript).toBeVisible({ timeout: 45_000 })
    console.log('STEP: transcript open')
    await target.screenshot({ path: path.join(evidence, 'archived-open.png') })

    // Give the control read time to settle either way, then assert the owner
    // resolution failure never paints.
    await target.waitForTimeout(5_000)
    const banner = target.getByText(/Session controls unavailable/)

    // AC5: the banner VISUALLY truncates (`truncate` span), so the id can only
    // be judged from the DOM text — capture it before asserting it is absent.
    const bannerTexts = await banner.allInnerTexts().catch(() => [] as string[])

    if (bannerTexts.length) {
      fs.writeFileSync(path.join(evidence, 'banner-text.txt'), bannerTexts.join('\n'))
      console.log('BANNER_TEXT: ' + JSON.stringify(bannerTexts))
    }

    await expect(banner, 'owner-resolution banner must not render').toHaveCount(0)
    console.log('STEP: no owner-resolution banner')

    // Composer status area — the surface the report said was broken.
    const composer = target.locator('[data-slot="composer-rich-input"]')
    await expect(composer).toBeVisible({ timeout: 15_000 })
    await composer.scrollIntoViewIfNeeded()
    await target.screenshot({ path: path.join(evidence, 'composer-status.png') })
    await target.screenshot({ path: path.join(evidence, 'full-window.png'), fullPage: true })

    const body = await target.locator('body').innerText()
    fs.writeFileSync(path.join(evidence, 'body.txt'), body)
    console.log('STEP: assertions complete')
  } finally {
    await app?.close().catch(() => undefined)
    await mock.close()

    const safeCopy = (from: string, to: string) => {
      try {
        if (fs.existsSync(from) && fs.statSync(from).isFile()) {
          fs.copyFileSync(from, to)
        }
      } catch {
        // evidence is best-effort — never mask the test's own failure
      }
    }

    for (const name of ['gateway.log', 'errors.log', 'desktop.log']) {
      safeCopy(path.join(sandbox.hermesHome, 'logs', name), path.join(evidence, name))
    }

    // The owning profile's own backend log, if the gateway wrote one there.
    const arthurLogs = path.join(arthurHome, 'logs')

    try {
      if (fs.existsSync(arthurLogs)) {
        for (const name of fs.readdirSync(arthurLogs)) {
          safeCopy(path.join(arthurLogs, name), path.join(evidence, `arthur-${name}`))
        }
      }
    } catch {
      // best-effort
    }

    try {
      spawnSync(
        python,
        [
          '-c',
          'import psutil,sys,os\nhome=os.path.realpath(sys.argv[1])\nowned=[]\nfor p in psutil.process_iter():\n    if p.pid==os.getpid(): continue\n    try:\n        if os.path.realpath(p.environ().get("HERMES_HOME") or os.sep)==home:\n            owned.append(p);p.terminate()\n    except (psutil.NoSuchProcess,psutil.AccessDenied): pass\n_,alive=psutil.wait_procs(owned,timeout=5)\nfor p in alive:\n    try:p.kill()\n    except psutil.NoSuchProcess:pass\n',
          sandbox.hermesHome
        ],
        { encoding: 'utf8', env: process.env }
      )
    } catch {
      // best-effort
    }

    try {
      sandbox.cleanup()
    } catch {
      // best-effort
    }
  }

  // Evidence trail: where the read went. `session.control.read` rides the
  // gateway socket, which desktop.log does not name per-method — the owning
  // profile's ROUTE line is the proof that the session window's requests (the
  // control read included) reached the backend that holds the session instead
  // of the window's ambient `default`.
  const desktopLog = path.join(evidence, 'desktop.log')

  const owningRoute = fs.existsSync(desktopLog)
    ? fs
        .readFileSync(desktopLog, 'utf8')
        .split('\n')
        .filter(line => line.includes('gateway-route') && line.includes('profile=arthur'))
    : []

  expect(owningRoute.length, 'the session window routed to the owning profile').toBeGreaterThan(0)
  console.log('OWNING_ROUTE_LINES:\n' + owningRoute.join('\n'))

  const logLines = ['desktop.log', 'gateway.log', 'arthur-gateway.log']
    .map(name => {
      const file = path.join(evidence, name)

      if (!fs.existsSync(file)) {
        return ''
      }

      return fs
        .readFileSync(file, 'utf8')
        .split('\n')
        .filter(line => line.includes('session.control.read'))
        .map(line => `${name}: ${line}`)
        .join('\n')
    })
    .filter(Boolean)
    .join('\n')

  console.log('CONTROL_READ_LOG_LINES:\n' + (logLines || '(none found)'))
})
