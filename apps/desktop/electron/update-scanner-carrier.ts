'use strict'

import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const UPDATE_SCANNER_RESOURCE_PATH = 'update-scanner/scan-venv-blockers.py'

interface ScannerCarrierResolution {
  defaultApp?: boolean
  resourcesPath?: string
  sourceRoot?: string
}

/** Resolve only candidate-owned scanner bytes, never a target-checkout module. */
export function resolveUpdateScannerCarrier(overrides: ScannerCarrierResolution = {}): string {
  const resourcesPath = overrides.resourcesPath ?? (process as any).resourcesPath
  const defaultApp = overrides.defaultApp ?? (process as any).defaultApp

  if (resourcesPath && defaultApp !== true) {
    return path.resolve(resourcesPath, UPDATE_SCANNER_RESOURCE_PATH)
  }

  const sourceRoot =
    overrides.sourceRoot ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

  return path.resolve(sourceRoot, 'resources', UPDATE_SCANNER_RESOURCE_PATH)
}

/** Keep each path as one argv element so Windows metacharacters remain inert. */
export function buildUpdateScannerArgv(
  canonicalRoot: string,
  carrierPath = resolveUpdateScannerCarrier()
): string[] {
  if (!path.isAbsolute(canonicalRoot) || !path.isAbsolute(carrierPath)) {
    throw new Error('scanner carrier and target root must be absolute paths')
  }

  return ['-I', carrierPath, '--root', canonicalRoot]
}
