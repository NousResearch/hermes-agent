import fs from 'node:fs'
import path from 'node:path'

function fileExists(filePath: string) {
  try {
    return fs.statSync(filePath).isFile()
  } catch {
    return false
  }
}

function cliWebDistForRoot(root: string | null | undefined) {
  return root ? path.join(root, 'hermes_cli', 'web_dist') : null
}

function pushUnique(candidates: string[], entry: string | null | undefined) {
  if (!entry || candidates.includes(entry)) {
    return
  }
  candidates.push(entry)
}

type ResolveDashboardWebDistOptions = {
  hermesRoot?: string | null
  activeHermesRoot: string
  sourceRepoRoot?: string | null
  isPackaged?: boolean
  dashboardOverride?: string | null
  hermesRootOverride?: string | null
  onWarning?: (message: string) => void
}

/**
 * Python dashboard/serve reads HERMES_WEB_DIST → hermes_cli/web_dist (Vite web/ SPA).
 * Never return the Electron renderer dist here — that path lives inside app.asar and 404s.
 */
function resolveDashboardWebDist({
  hermesRoot,
  activeHermesRoot,
  sourceRepoRoot,
  isPackaged = false,
  dashboardOverride,
  hermesRootOverride,
  onWarning
}: ResolveDashboardWebDistOptions) {
  const warn = onWarning ?? (() => {})

  if (dashboardOverride) {
    const resolved = path.resolve(dashboardOverride)
    if (fileExists(path.join(resolved, 'index.html'))) {
      return resolved
    }
    warn(`[web-dist] HERMES_DESKTOP_DASHBOARD_WEB_DIST set but index.html missing at ${resolved}`)
  }

  const candidates: string[] = []

  pushUnique(candidates, cliWebDistForRoot(hermesRoot))
  pushUnique(candidates, cliWebDistForRoot(hermesRootOverride))
  pushUnique(candidates, cliWebDistForRoot(activeHermesRoot))
  if (!isPackaged) {
    pushUnique(candidates, cliWebDistForRoot(sourceRepoRoot))
  }

  for (const cliDist of candidates) {
    if (fileExists(path.join(cliDist, 'index.html'))) {
      return cliDist
    }
  }

  const fallback = candidates[0] || path.join(activeHermesRoot, 'hermes_cli', 'web_dist')
  warn(
    `[web-dist] no dashboard web dist found (tried: ${candidates.join(', ') || '(none)'}). ` +
      `Build with: npm install --workspace web && npm run build -w web`
  )
  return fallback
}

export { cliWebDistForRoot, resolveDashboardWebDist }
