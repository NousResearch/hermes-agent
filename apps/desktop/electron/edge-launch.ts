import fs from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { spawn, execFile } from 'node:child_process'
import { promisify } from 'node:util'
import puppeteer from 'puppeteer-core'

const execute = promisify(execFile)
const pause = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

export async function ownedEdgeEndpoint(profile: string, executable: string) {
  profile = path.resolve(profile)
  executable = path.resolve(executable)
  const lines = (await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).trim().split(/\r?\n/)
  const port = Number(lines[0])
  if (!Number.isInteger(port) || port < 1 || port > 65535 || !/^\/devtools\/browser\/[a-zA-Z0-9-]+$/.test(lines[1] || '')) throw new Error('Invalid isolated Edge endpoint record.')
  const verification = await execute('C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', '$ErrorActionPreference="Stop"; $env:PSModulePath=Join-Path $PSHOME "Modules"; $owners=@(Get-NetTCPConnection -State Listen -LocalPort ([int]$env:ATHENA_EDGE_PORT) | Select-Object -ExpandProperty OwningProcess -Unique); if($owners.Count -ne 1){throw "Ambiguous endpoint owner"}; $proc=Get-CimInstance Win32_Process -Filter ("ProcessId="+$owners[0]); @{pid=$proc.ProcessId;path=$proc.ExecutablePath;command=$proc.CommandLine} | ConvertTo-Json -Compress'], { windowsHide: true, timeout: 10000, env: { ...process.env, ATHENA_EDGE_PORT: String(port) } })
  const owner = JSON.parse(verification.stdout)
  const command = String(owner.command)
  const profileArg = command.match(/(?:^|\s)--user-data-dir=(?:"([^"]+)"|([^\s"]+))(?=\s|$)/)
    || command.match(/(?:^|\s)"--user-data-dir=([^"]+)"(?=\s|$)/)
  const actualProfile = profileArg?.[1] || profileArg?.[2]
  if (typeof owner.path !== 'string' || owner.path.toLowerCase() !== executable.toLowerCase() || actualProfile?.toLowerCase() !== profile.toLowerCase()) throw new Error('Endpoint is not owned by the exact isolated Edge profile.')
  return { pid: Number(owner.pid), guest: /(?:^|\s)--guest(?:\s|$)/.test(command), endpoint: `ws://127.0.0.1:${port}${lines[1]}` }
}

/** Edge's Windows launcher may exit cleanly while a descendant owns the browser. */
export async function launchWindowsEdge(executable: string, headless: boolean, accountProfile?: string) {
  const profile = accountProfile ? path.resolve(accountProfile) : await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-edge-'))
  if (accountProfile) await fs.mkdir(profile, { recursive: true })
  const resolved = await fs.realpath(profile)
  if (!accountProfile) {
    await fs.mkdir(path.join(profile, 'Default'))
    await fs.writeFile(path.join(profile, 'Default', 'Preferences'), JSON.stringify({ signin: { allowed: false }, sync: { requested: false } }))
  }
  // This is the user-requested interactive browser, not a background helper.
  // Hiding its Windows GUI can leave the controlled page invisible to the user.
  const child = spawn(executable, [...(!accountProfile ? ['--guest', '--disable-sync'] : []), '--remote-debugging-port=0', '--remote-debugging-address=127.0.0.1', `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--disable-component-update', '--disable-default-apps', '--disable-extensions', ...(headless ? ['--headless=new'] : []), 'about:blank'], { windowsHide: headless, stdio: ['ignore', 'ignore', 'pipe'] })
  child.stderr.resume()
  let launchError: Error | undefined
  child.on('error', error => { launchError = error })
  const deadline = Date.now() + 30000
  let owner: Awaited<ReturnType<typeof ownedEdgeEndpoint>> | undefined
  while (Date.now() < deadline) {
    if (launchError) throw launchError
    try { owner = await ownedEdgeEndpoint(profile, executable); break } catch { /* Only inspect the new profile's endpoint; never retry launch. */ }
    if (child.exitCode !== null && child.exitCode !== 0) throw new Error(`Isolated Edge launcher exited with ${child.exitCode}.`)
    await pause(200)
  }
  if (!owner) throw new Error('Isolated Edge did not expose a verified endpoint; outcome is unknown. No replay.')
  if (!accountProfile && !owner.guest) throw new Error('The verified Edge process did not retain Guest mode. No attachment.')
  const browser = await puppeteer.connect({ browserWSEndpoint: owner.endpoint, defaultViewport: null, protocolTimeout: 30000, downloadBehavior: { policy: 'deny' } })
  const pid = owner.pid
  const cleanup = async () => {
    if (accountProfile) return // Deliberately persistent; never delete signed-in Athena profile data.
    // Delete only the exact newly-created profile, and only after its owning process has ended.
    for (let attempt = 0; attempt < 30; attempt++) {
      try { process.kill(pid, 0) } catch {
        if (path.dirname(resolved) === await fs.realpath(os.tmpdir()) && path.basename(resolved).startsWith('hermes-edge-') && await fs.realpath(profile) === resolved) {
          try { await fs.rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 }) } catch { /* Retain temp files if Windows still holds them; never change permissions. */ }
        }
        return
      }
      await pause(100)
    }
    // Retain the temp profile rather than deleting data from a live process.
  }
  browser.once('disconnected', () => { void cleanup().catch(() => undefined) })
  const verifySignedOut = async () => {
    const entries = await fs.readdir(profile, { withFileTypes: true })
    for (const entry of entries) {
      if (!entry.isDirectory() || !/^(Default|Guest Profile|Profile \d+)$/.test(entry.name)) continue
      let prefs: any
      try { prefs = JSON.parse(await fs.readFile(path.join(profile, entry.name, 'Preferences'), 'utf8')) } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue
        throw error
      }
      // Edge caches OS account metadata even in Guest mode. That alone is not browser sign-in.
      if (prefs.sync?.requested === true || prefs.sync?.has_setup_completed === true) throw new Error('Private Edge profile enabled sync. Preparation refused.')
    }
  }
  if (!accountProfile) {
    try { await verifySignedOut() } catch (error) { await browser.close(); throw error }
  }
  return { browser, pid, cleanup, verifySignedOut: accountProfile ? undefined : verifySignedOut }
}
