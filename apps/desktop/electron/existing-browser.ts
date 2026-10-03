import { PcDriverSession } from './pc-driver-session'
import { browserArguments } from './browser-session'
import { refreshLiveLifecycle, startBrowserLifecycle } from './browser-lifecycle'

type Driver = Pick<PcDriverSession, 'call' | 'close'>
interface ExistingRun { driver: Driver; pid: number; window: number; scope: string; label: string; active: boolean }

/** One explicitly approved native Chromium window per private granted driver. */
export class ExistingBrowsers {
  private windows = new Map<string, ExistingRun>()
  private targets = new Map<string, ExistingRun>()
  constructor(private command: string, private ownerKey: string, private createDriver = (command: string, args: string[]): Driver => new PcDriverSession(command, args)) {}
  get active() { return [...this.windows.values()].some(run => run.active) }
  scope(args: Record<string, unknown>) {
    const run = args.target_id ? this.targets.get(String(args.target_id)) : this.windows.get(`${args.pid}:${args.window_id}`)
    return run?.scope
  }
  handles(args: Record<string, unknown>) { return Boolean(this.scope(args)) }

  async prepare(args: Record<string, unknown>) {
    const key = `${args.pid}:${args.window_id}`
    if (this.windows.has(key)) throw new Error('This window already has an attachment in this conversation. Read its current state instead.')
    // Created only after native account-mode approval, never as part of private/PC access.
    const driver = this.createDriver(this.command, ['mcp', '--grant', 'existing-profile'])
    const scope = `existing_profile:${key}`
    const label = String(browserArguments('get_browser_state', {}, `${this.ownerKey}:${scope}`).session)
    const run: ExistingRun = { driver, pid: Number(args.pid), window: Number(args.window_id), scope, label, active: false }
    this.windows.set(key, run)
    try {
      await startBrowserLifecycle(driver, label)
      const reply = await driver.call('browser_prepare', { pid: run.pid, window_id: run.window, strategy: { kind: 'existing_profile' }, session: label })
      if (reply?.isError) { this.windows.delete(key); driver.close() }
      else run.active = true
      return reply
    } catch (error) { this.windows.delete(key); driver.close(); throw error }
  }

  async call(action: string, args: Record<string, unknown>) {
    if (action === 'browser_prepare' || (action !== 'get_browser_state' && !action.startsWith('browser_'))) throw new Error('Existing-browser attachments accept only browser reads and actions.')
    const run = args.target_id ? this.targets.get(String(args.target_id)) : this.windows.get(`${args.pid}:${args.window_id}`)
    if (!run || !run.active) throw new Error('Existing browser attachment is unavailable. No fallback.')
    if (action !== 'get_browser_state' && !args.target_id) throw new Error('Bind the exact approved browser window before acting.')
    const reply = await run.driver.call(action, { ...args, session: run.label })
    if (action === 'get_browser_state' && !reply?.isError) {
      const collect = (value: unknown) => {
        if (!value || typeof value !== 'object') return
        for (const [name, item] of Object.entries(value)) {
          if (name === 'target_id' && typeof item === 'string' && item.startsWith('bt-')) this.targets.set(item, run)
          else if (item && typeof item === 'object') collect(item)
        }
      }
      collect(reply.structuredContent)
      // Some driver versions provide the same structured envelope as a JSON text block.
      for (const block of reply.content || []) if (block.type === 'text') { try { collect(JSON.parse(block.text)) } catch { /* Human-readable snapshot. */ } }
    }
    return reply
  }

  async refresh() {
    for (const run of this.windows.values()) {
      if (!run.active) continue
      try {
        const implicit = await refreshLiveLifecycle(run.driver, {})
        const named = await refreshLiveLifecycle(run.driver, { session: run.label })
        if (!implicit || !named) run.active = false
      } catch { run.active = false }
    }
  }

  close() { for (const run of this.windows.values()) run.driver.close(); this.windows.clear(); this.targets.clear() }
}
