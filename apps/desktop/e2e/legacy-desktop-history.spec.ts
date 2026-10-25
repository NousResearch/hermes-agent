import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { expect, test } from './test'

// Pre-cutover Desktop rows as main writes them (source='desktop', profile stamped, optional
// Nous user_id), on an install with a second profile that has history of its own. A native
// HTTP grant covers one profile, so every cross-profile session list must still list them.
const ROWS = [
  { id: 'legacy-desktop-anon', source: 'desktop', userId: null },
  { id: 'legacy-desktop-nous', source: 'desktop', userId: 'nous:e2e-user' },
  { id: 'legacy-tui-control', source: 'tui', userId: null }
]

const NAMED_ROW = 'legacy-developer-desktop'

test('pre-cutover Desktop conversations remain visible and open across profiles', async () => {
  test.setTimeout(300_000)
  const sandbox = createSandbox('legacy-desktop-history')
  const mock = await startMockServer()
  const python = process.env.HERMES_DESKTOP_PYTHON!
  const root = path.resolve(import.meta.dirname, '../../..')
  const env = buildAppEnv(sandbox, { HERMES_DESKTOP_PYTHON: python, TIRITH_ENABLED: 'false' })
  const multiplex = process.env.E2E_MULTIPLEX === '1'
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, multiplex ? 'gateway:\n  multiplex_profiles: true\n' : undefined)
  writeEnvFile(sandbox.hermesHome)
  const named = path.join(sandbox.hermesHome, 'profiles', 'developer')
  fs.mkdirSync(named, { recursive: true })
  writeMockProviderConfig(named, mock.url)
  writeEnvFile(named)

  const seed = spawnSync(python, ['-c', `
import json, os, sys
from pathlib import Path
from hermes_state import SessionDB

def seed(db_path, rows, profile):
    with SessionDB(db_path=db_path) as db:
        for r in rows:
            db.create_session(r['id'], source=r['source'], model='e2e-mock', cwd=os.environ['HOME'], user_id=r['userId'])
            db.set_session_title(r['id'], 'Retained ' + r['id'])
            db.append_message(r['id'], 'user', 'Historical question ' + r['id'])
            db.append_message(r['id'], 'assistant', 'Historical answer ' + r['id'])
            db._conn.execute('UPDATE sessions SET profile_name=? WHERE id=?', (profile, r['id']))
        db._conn.commit()

home = Path(os.environ['HERMES_HOME'])
seed(home / 'state.db', json.loads(sys.argv[1]), 'default')
seed(home / 'profiles' / 'developer' / 'state.db', [{'id': sys.argv[2], 'source': 'desktop', 'userId': None}], 'developer')
`, JSON.stringify(ROWS), NAMED_ROW], { cwd: root, env: { ...env, PYTHONPATH: root }, encoding: 'utf8' })

  expect(seed.status, seed.stderr).toBe(0)

  // One cron job in each profile's own store, for the all-profiles cron list.
  for (const [home, name] of [[sandbox.hermesHome, 'default-job'], [named, 'developer-job']]) {
    const job = spawnSync(python, ['-c', 'import sys\nfrom cron.jobs import create_job\ncreate_job("e2e", "every 1h", name=sys.argv[1])', name], {
      cwd: root, env: { ...env, HERMES_HOME: home, PYTHONPATH: root }, encoding: 'utf8'
    })

    expect(job.status, job.stderr).toBe(0)
  }

  // An existing install has a startup profile; without one the first-launch migration guesses
  // from the newest state.db, which here is the named profile's.
  fs.writeFileSync(path.join(sandbox.userDataDir, 'active-profile.json'), JSON.stringify({ profile: 'default' }))
  let app: Awaited<ReturnType<typeof launchDesktop>>['app'] | undefined
  const evidence = process.env.HERMES_E2E_EVIDENCE || path.join(sandbox.root, 'evidence')
  fs.mkdirSync(evidence, { recursive: true })
  const results: Record<string, { listed: boolean; opened: boolean; error?: string }> = {}
  let probes: Record<string, unknown> = {}
  const views: Record<string, string[]> = {}
  let resumeFailures = -1

  try {
    const launched = await launchDesktop(env)
    app = launched.app
    const page = launched.page
    const sidebar = page.locator('[data-slot="sidebar"]')

    try {
      await expect(page.getByRole('button', { name: /developer/ }).first()).toBeVisible({ timeout: 90_000 })

      // The cross-profile lists the sidebar and Sessions view read, through main's transport.
      probes = await page.evaluate(async () => {
        const api = (window as any).hermesDesktop.api
        const out: Record<string, unknown> = {}
        const q = 'limit=40&offset=0&min_messages=1&archived=exclude&order=recent'
        // The same all-profiles read as the renderer sends it on its registry route.
        const registry = { profile: 'default', connectionId: 'local' }

        // One probe: its result, or the error text, under `name`.
        const probe = async (name: string, read: () => Promise<unknown>) => {
          try {
            out[name] = await read()
          } catch (e) {
            out[name] = 'ERR ' + String((e as Error)?.message ?? e).slice(0, 200)
          }
        }

        const sessionIds = (r: any) => {
          const rows = r?.recents ? r.recents.sessions : r?.sessions

          return { ids: (rows ?? []).map((x: any) => `${x.profile}:${x.id}`).sort(), errors: r?.errors ?? [] }
        }

        const previewProfiles = (r: any) => {
          const profiles = new Set<string>()

          for (const project of r?.projects ?? []) {
            for (const session of project.previewSessions ?? []) {
              profiles.add(session.profile)
            }
          }

          return { profiles: [...profiles].sort(), errors: r?.errors ?? [] }
        }

        const cronNames = (r: any) => (Array.isArray(r) ? r : []).map((job: any) => `${job.profile}:${job.name}`).sort()

        // The cross-profile lists the sidebar and Sessions view read, through main's transport.
        const sidebarAll = '/api/profiles/sessions/sidebar?recents_profile=all&recents_limit=40&cron_limit=5&messaging_limit=5'

        const reads: [string, Record<string, unknown>, (r: any) => unknown][] = [
          ['sidebar:default', { path: '/api/profiles/sessions/sidebar?recents_profile=default&recents_limit=40&cron_limit=5&messaging_limit=5' }, sessionIds],
          ['sidebar:developer', { path: '/api/profiles/sessions/sidebar?recents_profile=developer&recents_limit=40&cron_limit=5&messaging_limit=5' }, sessionIds],
          ['sidebar:all', { path: sidebarAll }, sessionIds],
          ['sessions:all', { path: `/api/profiles/sessions?${q}&profile=all` }, sessionIds],
          ['sidebar:all:registry', { path: sidebarAll, ...registry }, sessionIds],
          // The all-profiles Projects view and PR badges: which profiles' sessions each read covers.
          ['projects:all', { path: '/api/profiles/projects/tree?preview_limit=10' }, previewProfiles],
          ['projects:all:registry', { path: '/api/profiles/projects/tree?preview_limit=10', ...registry }, previewProfiles],
          // The all-profiles cron list, on both routes.
          ['cron:all', { path: '/api/cron/jobs?profile=all' }, cronNames],
          ['cron:all:registry', { path: '/api/cron/jobs?profile=all', ...registry }, cronNames],
          ['prs:all', { path: '/api/profiles/sessions/pull-requests', method: 'POST', body: { ids: ['legacy-desktop-anon'] } },
            (r: any) => ({ scanned: r?.scanned ?? null })]
        ]

        for (const [name, request, shape] of reads) {
          await probe(name, async () => shape(await api(request)))
        }

        return out
      })
      console.log('API_PROBES: ' + JSON.stringify(probes))

      await expect(sidebar.getByText(/Retained legacy-/).first()).toBeVisible({ timeout: 30_000 }).catch(() => undefined)
      await page.screenshot({ path: path.join(evidence, 'sidebar.png') })

      for (const row of [...ROWS, { id: NAMED_ROW }]) {
        if (row.id === NAMED_ROW) {
          await page.getByRole('button', { name: /developer/ }).first().click()
        }

        const entry = sidebar.getByText('Retained ' + row.id, { exact: true })
        await entry.waitFor({ state: 'visible', timeout: 30_000 }).catch(() => undefined)
        const listed = await entry.isVisible()
        results[row.id] = { listed, opened: false }

        if (!listed) {
          continue
        }

        await entry.click()
        const answer = page.locator('[data-slot="aui_thread-viewport"]').getByText('Historical answer ' + row.id)
        const failure = page.getByText("Couldn't open this session")
        await expect(answer.or(failure).first()).toBeVisible({ timeout: 45_000 }).catch(() => undefined)
        results[row.id].opened = await answer.isVisible()

        if (!results[row.id].opened) {
          results[row.id].error = (await page.locator('body').innerText()).match(/Couldn't open this session\s*\n?\s*(\S+)/)?.[1] ?? 'no transcript'
        }

        await page.screenshot({ path: path.join(evidence, `open-${row.id}.png`) })
      }

      // The rail's home pill: back to default, then its all-profiles view, then default again.
      // Every profile's history is listed, and leaving the all view keeps default's own.
      const listedIn = async (ids: string[]) => {
        const seen: string[] = []

        for (const id of ids) {
          const entry = sidebar.getByText('Retained ' + id, { exact: true }).first()
          await entry.waitFor({ state: 'visible', timeout: 15_000 }).catch(() => undefined)

          if (await entry.isVisible()) {
            seen.push(id)
          }
        }

        return seen
      }

      const rail = page.locator('[data-slot="profile-rail"]')
      await rail.getByRole('button', { name: /^Switch to / }).first().click()
      await rail.getByRole('button', { name: 'Show all profiles' }).first().click()
      views.all = await listedIn([...ROWS.map(r => r.id), NAMED_ROW])
      await page.screenshot({ path: path.join(evidence, 'view-all.png') })
      await rail.getByRole('button', { name: /^Switch to / }).first().click()
      views.default = await listedIn(ROWS.map(r => r.id))
      await page.screenshot({ path: path.join(evidence, 'view-default.png') })

      // The Bots plugin's hide write as the SDK sends it: the owner named only in the body, from
      // the window, now back on default. An unopened pre-upgrade row is not editable over HTTP at all
      // (only its first resume adopts it), so this runs on the developer row opened above.
      probes['bots:hide'] = await page.evaluate(async () => {
        const api = (window as any).hermesDesktop.api
        const path = '/api/sessions/legacy-developer-desktop'

        try {
          const snapshot = await api({ path: `${path}/mutation-snapshot?profile=developer` })

          const identity = {
            expected_generation: snapshot?.runtime_generation,
            expected_revision: snapshot?.runtime_revision,
            request_id: crypto.randomUUID()
          }

          const r = await api({ path, method: 'PATCH', body: { hidden: false, profile: 'developer', ...identity } })

          return { ok: r?.ok ?? null }
        } catch (e) {
          return 'ERR ' + String((e as Error)?.message ?? e).slice(0, 200)
        }
      })
      console.log('BOTS_HIDE: ' + JSON.stringify(probes['bots:hide']))
    } finally {
      console.log('LEGACY_DESKTOP_RESULTS: ' + JSON.stringify(results))
      console.log('LEGACY_DESKTOP_VIEWS: ' + JSON.stringify(views))
      fs.writeFileSync(path.join(evidence, 'results.json'), JSON.stringify({ results, views, probes }, null, 2))

      for (const name of ['gateway.log', 'errors.log', 'desktop.log']) {
        const log = path.join(sandbox.hermesHome, 'logs', name)

        if (fs.existsSync(log)) {
          fs.copyFileSync(log, path.join(evidence, name))
        }
      }

      // Opening a row resumes it; every one, Nous-stamped included, must be continuable.
      const desktopLog = path.join(sandbox.hermesHome, 'logs', 'desktop.log')
      resumeFailures = fs.existsSync(desktopLog) ? (fs.readFileSync(desktopLog, 'utf8').match(/Resume failed/g) ?? []).length : -1
      console.log('LEGACY_DESKTOP_RESUME_FAILURES: ' + resumeFailures)
    }
  } finally {
    await app?.close().catch(() => undefined)
    await mock.close()
    spawnSync(python, ['-c', `
import psutil,sys,os
home=os.path.realpath(sys.argv[1])
owned=[]
for p in psutil.process_iter():
    if p.pid==os.getpid(): continue
    try:
        if os.path.realpath(p.environ().get('HERMES_HOME') or os.sep)==home:
            owned.append(p);p.terminate()
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
_,alive=psutil.wait_procs(owned,timeout=5)
for p in alive:
    try:p.kill()
    except psutil.NoSuchProcess:pass
`, sandbox.hermesHome], { encoding: 'utf8', env: process.env })
    sandbox.cleanup()
  }

  const everyRow = [...ROWS.map(r => `default:${r.id}`), `developer:${NAMED_ROW}`].sort()
  expect.soft(probes['sidebar:all'], 'sidebar:all').toEqual({ ids: everyRow, errors: [] })
  expect.soft(probes['sidebar:all:registry'], 'sidebar:all:registry').toEqual({ ids: everyRow, errors: [] })
  const everyProfile = { profiles: ['default', 'developer'], errors: [] }
  expect.soft(probes['projects:all'], 'projects:all').toEqual(everyProfile)
  expect.soft(probes['projects:all:registry'], 'projects:all:registry').toEqual(everyProfile)
  expect.soft(probes['prs:all'], 'prs:all').toEqual({ scanned: ['legacy-desktop-anon'] })
  const everyJob = ['default:default-job', 'developer:developer-job']
  expect.soft(probes['cron:all'], 'cron:all').toEqual(everyJob)
  expect.soft(probes['cron:all:registry'], 'cron:all:registry').toEqual(everyJob)
  expect.soft(probes['bots:hide'], 'bots:hide').toEqual({ ok: true })
  expect.soft(probes['sessions:all'], 'sessions:all').toEqual({ ids: everyRow, errors: [] })
  expect.soft(probes['sidebar:default'], 'sidebar:default').toEqual({ ids: ROWS.map(r => `default:${r.id}`).sort(), errors: [] })
  expect.soft(probes['sidebar:developer'], 'sidebar:developer').toEqual({ ids: [`developer:${NAMED_ROW}`], errors: [] })

  for (const row of [...ROWS, { id: NAMED_ROW }]) {
    expect.soft(results[row.id], row.id).toEqual({ listed: true, opened: true })
  }

  expect.soft(views.all, 'all-profiles view').toEqual([...ROWS.map(r => r.id), NAMED_ROW])
  expect.soft(views.default, 'default after all-profiles view').toEqual(ROWS.map(r => r.id))
  expect.soft(resumeFailures, 'resume failures').toBe(0)
})
