import type { ElectronApplication, Page } from '@playwright/test'

import type { NativeProcess } from '../../../tests-js/scripts/desktop-smoke-process.ts'

export interface UpdateWindowIdentity {
  hermesRoot: string
}

export interface UpdateWindowChatOptions {
  mockUrl: string
  outDir: string
  expectCommit: string
  origin: 'source' | 'bundled'
  root: string
  executable: string
  userData: string
}

export interface UpdateWindowProcess {
  executable: string
  resources: string
  userData: string
}

export function assertUpdateWindowProcess(
  running: UpdateWindowProcess,
  options: Pick<UpdateWindowChatOptions, 'executable' | 'origin' | 'root' | 'userData'>,
): void

export function assertUpdateWindowBackendOrigin(
  backend: NativeProcess,
  identity: UpdateWindowIdentity,
  root: string,
  origin: 'source' | 'bundled',
): void

export function runUpdateWindowChat(
  app: ElectronApplication,
  page: Page,
  options: UpdateWindowChatOptions,
): Promise<void>
