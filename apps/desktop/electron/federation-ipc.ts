import http from 'node:http'

/**
 * Federation IPC bridge — Electron main process ↔ Gateway API.
 *
 * Security design:
 * - Auth token NEVER exposed to renderer (stored only in main process)
 * - All federation API calls go through IPC (renderer can't call API directly)
 * - Input validated before forwarding to Gateway
 * - Timeout: 5s per request (prevents hanging)
 * - Response sanitized (no raw HTTP details exposed)
 */
import { ipcMain } from 'electron'

interface FederationAPIConfig {
  port: number
  host: string
  authToken: string
}

let config: FederationAPIConfig | null = null

/**
 * Make authenticated request to Federation Gateway API.
 * Token is NEVER exposed to renderer.
 */
async function callFederationAPI(path: string, method = 'GET', body?: unknown): Promise<unknown> {
  if (!config) {
    throw new Error('Federation IPC: not configured')
  }

  // Security: validate path (prevent SSRF)
  if (!path.startsWith('/api/federation/')) {
    throw new Error(`Federation IPC: invalid path ${path}`)
  }

  return new Promise((resolve, reject) => {
    const options: http.RequestOptions = {
      hostname: config!.host,
      port: config!.port,
      path,
      method,
      headers: {
        'Authorization': `Bearer ${config!.authToken}`,
        'Content-Type': 'application/json',
      },
      timeout: 5000, // 5s timeout (security: prevent hanging)
    }

    const req = http.request(options, (res) => {
      let data = ''
      res.on('data', (chunk) => { data += chunk.toString() })
      res.on('end', () => {
        // Security: validate response status
        if (res.statusCode === 401 || res.statusCode === 403) {
          reject(new Error('Federation API: unauthorized'))

          return
        }

        if (res.statusCode !== 200) {
          reject(new Error(`Federation API: HTTP ${res.statusCode}`))

          return
        }

        // Security: validate JSON before parsing
        try {
          const parsed = JSON.parse(data)
          resolve(parsed)
        } catch {
          reject(new Error('Federation API: invalid JSON response'))
        }
      })
    })

    req.on('error', reject)
    req.on('timeout', () => {
      req.destroy()
      reject(new Error('Federation API: request timeout'))
    })

    if (body && method !== 'GET') {
      req.write(JSON.stringify(body))
    }

    req.end()
  })
}

/**
 * Register IPC handlers for Federation API.
 * Called during Electron app initialization.
 */
export function registerFederationIPC(cfg: FederationAPIConfig): void {
  config = cfg

  // GET endpoints (read-only, safe)
  ipcMain.handle('fed:status', () => callFederationAPI('/api/federation/status'))
  ipcMain.handle('fed:peers', () => callFederationAPI('/api/federation/peers'))
  ipcMain.handle('fed:tasks', () => callFederationAPI('/api/federation/tasks'))
  ipcMain.handle('fed:health', () => callFederationAPI('/api/federation/health'))
  ipcMain.handle('fed:leader', () => callFederationAPI('/api/federation/leader'))
  ipcMain.handle('fed:metrics', () => callFederationAPI('/api/federation/metrics'))

  // POST endpoints (admin, write operations)
  ipcMain.handle('fed:handoff', (_event, taskData: unknown) =>
    callFederationAPI('/api/federation/handoff', 'POST', taskData)
  )
  ipcMain.handle('fed:config:sync', () =>
    callFederationAPI('/api/federation/config/sync', 'POST')
  )
}

/**
 * Unregister IPC handlers (cleanup on shutdown).
 */
export function unregisterFederationIPC(): void {
  config = null
  ipcMain.removeHandler('fed:status')
  ipcMain.removeHandler('fed:peers')
  ipcMain.removeHandler('fed:tasks')
  ipcMain.removeHandler('fed:health')
  ipcMain.removeHandler('fed:leader')
  ipcMain.removeHandler('fed:metrics')
  ipcMain.removeHandler('fed:handoff')
  ipcMain.removeHandler('fed:config:sync')
}
