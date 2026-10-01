// drive-update.cjs — launch the INSTALLED Hermes.exe (real Electron desktop
// app) under Playwright's Electron driver and perform the update the way a
// user does: Settings -> About -> "Update now". Screenshots at every step.
//
// Run the current CI checkout's entrypoint with its locked driver deps:
//
//   node <this file> <path-to-Hermes.exe> <proof-dir> <old-sha> [--native-handoff]
// --native-handoff leaves the native UIA caller in charge of clicking Update.
//
// Exit codes: 0 = update hand-off started and the app quit (the detached
// updater takes it from there — the PowerShell driver polls for the result);
// 1 = any step failed. The driver treats nonzero as leg failure.
//
// This intentionally does NOT call any store/bridge function directly: only
// real clicks on the real UI, so a regression in the button, the About
// panel, the overlay, or the renderer->main bridge fails the test.

const path = require('node:path')
const fs = require('node:fs')

const { _electron } = require('@playwright/test')
const { prepareWindowForInput } = require('./window-input.cjs')
const { observeProcessClose } = require('./process-close.cjs')
const { pickAppWindow, openAbout, readManualUpdateCommand, waitForUpdate } = require('./update-ui.cjs')

const exePath = process.argv[2]
const proofDir = process.argv[3]
const oldSha = process.argv[4]
const nativeHandoff = process.argv[5] === '--native-handoff'

if (!exePath || !proofDir || !oldSha || !process.env.HERMES_E2E_MOCK_URL) {
  console.error('usage: node drive-update.cjs <Hermes.exe> <proof-dir> <old-sha> [--native-handoff]; HERMES_E2E_MOCK_URL required')
  process.exit(1)
}

fs.mkdirSync(proofDir, { recursive: true })

function log(msg) {
  console.log(`[drive-update] ${new Date().toISOString()} ${msg}`)
}

async function shot(page, name) {
  const file = path.join(proofDir, `${name}.png`)

  try {
    await page.screenshot({ path: file })
    log(`screenshot: ${file}`)
  } catch (err) {
    log(`screenshot ${name} failed: ${err.message}`)
  }
}

// Hard ceiling so a hung renderer can't wedge the CI job; the driver's own
// step timeout is the real guard, this is belt-and-braces.
const KILL_AFTER_MS = 15 * 60 * 1000
const killer = setTimeout(() => {
  console.error('[drive-update] global timeout — aborting')
  process.exit(1)
}, KILL_AFTER_MS)
killer.unref()

async function main() {
  const { runUpdateWindowChat } = await import('./update-window-chat.mjs')
  const { isolateUpdateWindowEnvironment, isolatedElectronArgs, updateWindowEnvironment } = await import('./smoke-env.mjs')
  const origin = nativeHandoff ? 'bundled' : 'source'
  const root = nativeHandoff ? path.join(path.dirname(exePath), 'resources', 'agent-payload') : path.join(process.env.HERMES_HOME, 'hermes-agent')
  const launchEnv = isolateUpdateWindowEnvironment(updateWindowEnvironment(process.env, root, origin))
  const userData = launchEnv.HERMES_DESKTOP_USER_DATA_DIR
  log(`launching ${exePath}`)

  const app = await _electron.launch({
    executablePath: exePath,
    args: isolatedElectronArgs(['--disable-gpu', '--no-sandbox', '--force-renderer-accessibility'], userData),
    cwd: path.dirname(exePath),
    // Inherit the driver's env: HERMES_HOME (isolated install) and
    // GIT_CONFIG_GLOBAL (URL redirect to the staged serve repo) MUST reach
    // the main process so its update check fetches from the staged repo.
    env: launchEnv,
    timeout: 120_000
  })
  const child = app.process()

  const waitForProcessClose = observeProcessClose(child)
  // On Windows Playwright's child is a shell wrapper, not Hermes.exe.
  const appPid = await app.evaluate(() => process.pid)
  log(`launched Electron pid=${appPid}`)

  const page = await pickAppWindow(app, log)

  await prepareWindowForInput(app, page)
  log('[zoom] app window prepared at 100%')

  await runUpdateWindowChat(app, page, {
    mockUrl: process.env.HERMES_E2E_MOCK_URL, outDir: proofDir,
    expectCommit: oldSha,
    origin, root, executable: exePath, userData,
  })
  await shot(page, '01-app-booted')

  if (nativeHandoff) {
    fs.writeFileSync(path.join(proofDir, 'old-chat-ready.json'), JSON.stringify({ pid: appPid, exe: exePath, oldSha }) + '\n')
    await waitForProcessClose(15 * 60 * 1000)
    log('native OLD process closed; the OS updater owns relaunch')
    return
  }

  await openAbout(page, { log, shot, prepare: () => prepareWindowForInput(app, page) })
  const updateNow = await waitForUpdate(page, { log, shot })

  let appClosed = false
  app.on('close', () => {
    appClosed = true
  })

  // ── The click under test ──────────────────────────────────────────────
  await updateNow.click()
  log('clicked: Update now')

  // The "Updating Hermes — this window will close" overlay should appear,
  // then the app quits (hand-off dwell). Screenshot the overlay while the
  // window is still alive.
  // The app can close during the dwell. This wait must outlive its page.
  await new Promise(resolve => setTimeout(resolve, 1200))
  await shot(page, '05-updating-overlay')

  const manualCommand = await readManualUpdateCommand(page)
  if (manualCommand) {
    fs.writeFileSync(
      path.join(proofDir, 'manual-update.json'),
      `${JSON.stringify({ command: manualCommand, oldSha }, null, 2)}\n`,
    )
    log(`OLD requires the manual update path: ${manualCommand}`)
    await app.close()
    await waitForProcessClose()
    return 42
  }

  // ── Wait for the hand-off to take over ────────────────────────────────
  // Clicking Update now spawns the detached updater (desktop-update.ps1 or
  // the staged binary), which claims HERMES_HOME/.hermes-update-in-progress
  // and then the desktop quits. We do NOT rely on Playwright's app 'close'
  // event: when the app self-quits for the hand-off that event is
  // unreliable (attempt 8 timed out on it even though the hand-off log
  // proved the desktop had exited and `hermes update` was already running).
  //
  // The authoritative "hand-off started" signal is the marker file (or the
  // result JSON, if the whole update finished fast). Poll for either, and
  // also accept a genuine app close. Any one is success — the PowerShell
  // driver owns asserting the update's OUTCOME (sha, marker cleanup,
  // relaunch) after we return.
  const hermesHome = process.env.HERMES_HOME
  const markerPath = hermesHome ? path.join(hermesHome, '.hermes-update-in-progress') : null
  const resultPath = hermesHome ? path.join(hermesHome, '.hermes-update-result.json') : null

  const handoffDeadline = Date.now() + 150_000
  let handoffStarted = false

  while (Date.now() < handoffDeadline) {
    if (markerPath && fs.existsSync(markerPath)) {
      log('hand-off marker present — updater has taken over')
      handoffStarted = true
      break
    }
    if (resultPath && fs.existsSync(resultPath)) {
      log('update result JSON already present — updater finished fast')
      handoffStarted = true
      break
    }
    if (appClosed) {
      log('app closed — hand-off in progress')
      handoffStarted = true
      break
    }
    // Secondary: if the renderer window is gone, evaluate throws.
    try {
      await page.evaluate(() => true)
    } catch {
      log('renderer window gone — app quit for hand-off')
      handoffStarted = true
      break
    }
    await new Promise(r => setTimeout(r, 2000))
  }

  if (!handoffStarted) {
    await shot(page, 'ERROR-no-handoff')
    throw new Error('no hand-off within 150s of Update now (no marker, no result, app still alive)')
  }

  // A marker appears before Electron exits. Exiting this driver at that point
  // lets Playwright taskkill the entire tree, including the detached updater.
  await waitForProcessClose()
  log('Electron process closed — detached updater owns the rest')
}

main()
  .then(code => process.exit(code || 0))
  .catch(err => {
    console.error(`[drive-update] FAILED: ${err.message}`)
    process.exit(1)
  })
