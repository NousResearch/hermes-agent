import crypto from 'node:crypto'
import net from 'node:net'
import { spawn, type ChildProcess } from 'node:child_process'
import { webUrl } from './owned-browser'

const elementKey = 'element-6066-11e4-a52e-4f735466cecf'
const result = (data: unknown) => ({ content: [{ type: 'text', text: JSON.stringify(data) }], structuredContent: data })
type Ref = { element: Record<string, string>; tag: string; type: string; href: string; download: boolean }

/** Safari's own isolated WebDriver window. Never attaches to personal Safari tabs. */
export class SafariBrowser {
  private child?: ChildProcess
  private endpoint = ''
  private session = ''
  private target = ''
  private tab = ''
  private refs = new Map<string, Ref>()
  private snapshotUrl = ''
  private dialog?: { id: string; text: string }
  private closed = false
  get active() { return Boolean(this.session && this.child && this.child.exitCode === null && !this.closed) }
  owns(target: unknown) { return this.active && target === this.target }

  private async request(method: string, route: string, body?: unknown, timeout = 30000): Promise<any> {
    const response = await fetch(this.endpoint + route, { method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(timeout) })
    const data = await response.json() as { value?: any }
    if (!response.ok || data.value?.error) throw new Error(`Safari ${data.value?.error || response.status}: ${data.value?.message || 'WebDriver request failed'}`)
    return data.value
  }
  private command(method: string, route: string, body?: unknown) {
    if (!this.active) throw new Error('Safari automation access ended. No fallback.')
    return this.request(method, `/session/${this.session}${route}`, body)
  }
  private script(script: string, args: unknown[] = []) { return this.command('POST', '/execute/sync', { script, args }) }

  async prepare() {
    if (process.platform !== 'darwin') throw new Error('Safari automation is available only on macOS.')
    if (this.closed || this.session) throw new Error('Safari is already prepared or revoked. Do not create another session.')
    const reservation = net.createServer()
    await new Promise<void>((resolve, reject) => { reservation.once('error', reject); reservation.listen(0, '127.0.0.1', resolve) })
    const port = (reservation.address() as net.AddressInfo).port
    await new Promise<void>((resolve, reject) => reservation.close(error => error ? reject(error) : resolve()))
    if (this.closed) throw new Error('Safari access was revoked before launch.')
    this.endpoint = `http://127.0.0.1:${port}`
    this.child = spawn('/usr/bin/safaridriver', ['-p', String(port)], { stdio: 'ignore' })
    let startupError: Error | undefined
    this.child.once('error', error => { startupError = error })
    this.child.once('exit', () => { this.refs.clear() })
    try {
      let ready = false
      for (let attempt = 0; attempt < 50; attempt++) {
        if (startupError || this.child.exitCode !== null) throw startupError || new Error('The dedicated Safari driver exited.')
        try { await this.request('GET', '/status', undefined, 500); ready = true; break } catch { await new Promise(resolve => setTimeout(resolve, 100)) }
      }
      if (!ready) throw new Error('The dedicated Safari driver did not start.')
      // Safari refuses concurrent automation; never terminate another driver's session.
      const created = await this.request('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari', unhandledPromptBehavior: 'ignore' } } })
      if (this.child.exitCode !== null || !created?.sessionId || created.capabilities?.browserName?.toLowerCase() !== 'safari') throw new Error('Safari did not confirm the requested automation session.')
      this.session = created.sessionId
      await this.command('POST', '/timeouts', { implicit: 0, script: 10000, pageLoad: 25000 })
      this.target = 'saf-' + crypto.randomUUID()
      this.tab = 'tab-' + crypto.randomUUID()
      return result({ status: 'prepared', browser: 'safari', profile_mode: 'isolated_new', isolated_profile: true, persistent_profile: false, private_storage_context: true, target_id: this.target, tab_id: this.tab, next_call: { name: 'desktop_browser_read', arguments: { target_id: this.target, tab_id: this.tab } } })
    } catch (error) { await this.close(); throw error }
  }

  async call(action: string, args: Record<string, unknown>) {
    if (!this.owns(args.target_id) || args.tab_id !== this.tab) throw new Error('Safari target belongs to another conversation or ended session. No fallback.')
    if (action === 'browser_navigate') {
      const url = webUrl(args.url)
      this.refs.clear()
      await this.command('POST', '/url', { url })
      const observed = await this.command('GET', '/url')
      webUrl(observed)
      return result({ status: 'navigated', url: observed, title: await this.command('GET', '/title') })
    }
    if (action === 'get_browser_state') {
      const url = await this.command('GET', '/url')
      webUrl(url)
      const snapshot = await this.script(`return {title:document.title,text:(document.body?.innerText||'').slice(0,18000),controls:Array.from(document.querySelectorAll('a,button,input,textarea,select,[role="button"],[contenteditable="true"]')).filter(e=>e.getClientRects().length).slice(0,200).map(e=>({element:e,tag:e.tagName.toLowerCase(),type:e.type||'',href:e.href||'',download:e.hasAttribute('download'),text:(e.getAttribute('aria-label')||e.innerText||e.placeholder||'').slice(0,240),value:e.type==='password'?undefined:e.value}))}`)
      this.refs.clear()
      this.snapshotUrl = url
      const generation = 'p' + crypto.randomUUID().replaceAll('-', '')
      const controls = snapshot.controls.map((control: any, index: number) => {
        if (!control.element?.[elementKey]) throw new Error('Safari returned an invalid element identity.')
        const ref = `${generation}:${index}`
        this.refs.set(ref, control)
        const { element, ...description } = control
        return { ref, ...description }
      })
      const state = result({ target_id: this.target, tab_id: this.tab, url, title: snapshot.title, text: snapshot.text, controls })
      if (args.include_screenshot === true) {
        const data = await this.command('GET', '/screenshot')
        if (await this.command('GET', '/url') !== url) throw new Error('Safari changed during capture; read fresh state.')
        return { ...state, content: [...state.content, { type: 'image', mimeType: 'image/png', data }] }
      }
      return state
    }
    if (action === 'browser_dialog') {
      const text = await this.command('GET', '/alert/text')
      if (args.action === 'inspect') {
        this.dialog = { id: 'dialog-' + crypto.randomUUID(), text }
        return result({ dialog_id: this.dialog.id, text })
      }
      if (!this.dialog || this.dialog.id !== args.dialog_id || this.dialog.text !== text) throw new Error('Safari dialog changed; inspect it again before acting.')
      if (!['accept', 'dismiss'].includes(String(args.action))) throw new Error('Unsupported Safari dialog action.')
      this.dialog = undefined
      if (args.prompt_text !== undefined) {
        if (args.action !== 'accept' || typeof args.prompt_text !== 'string') throw new Error('Prompt text requires an explicit accept action.')
        await this.command('POST', '/alert/text', { text: args.prompt_text })
      }
      await this.command('POST', `/alert/${args.action}`, {})
      this.refs.clear()
      return result({ status: 'dialog_handled' })
    }
    if (!['browser_click', 'browser_type'].includes(action)) throw new Error('This Safari prototype supports navigation, reads, clicks, typing and dialogs. Pointer gestures are unavailable.')
    const ref = this.refs.get(String(args.ref))
    if (!ref) throw new Error('Stale Safari element ref. Read fresh state before acting.')
    if (await this.command('GET', '/url') !== this.snapshotUrl) { this.refs.clear(); throw new Error('Safari navigated; read fresh state before acting.') }
    if (ref.type === 'file' || ref.download) throw new Error('File upload/download controls are not supported.')
    if (ref.href) webUrl(ref.href)
    const element = encodeURIComponent(ref.element[elementKey])
    this.refs.clear() // Consume the snapshot before a potentially uncertain side effect.
    if (action === 'browser_click') await this.command('POST', `/element/${element}/click`, {})
    else {
      if (typeof args.text !== 'string' || args.text.length > 64000 || !['input', 'textarea'].includes(ref.tag)) throw new Error('Safari typing requires a bounded text string and an editable input or textarea.')
      if (args.replace === true) await this.command('POST', `/element/${element}/clear`, {})
      await this.command('POST', `/element/${element}/value`, { text: args.text })
    }
    return result({ status: 'delivered', verification_required: true })
  }

  async close() {
    if (this.closed) return
    this.closed = true
    this.refs.clear()
    if (this.session) try { await this.request('DELETE', `/session/${this.session}`, undefined, 3000) } catch { /* Teardown only; never retry an action. */ }
    this.session = ''
    this.child?.kill('SIGTERM')
  }
}
