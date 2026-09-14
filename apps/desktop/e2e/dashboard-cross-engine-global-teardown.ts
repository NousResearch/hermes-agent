import fs from 'node:fs'

import { CROSS_ENGINE_HERMES_HOME } from './dashboard-cross-engine-home'

export default async function globalTeardown(): Promise<void> {
  try {
    fs.rmSync(CROSS_ENGINE_HERMES_HOME, { recursive: true, force: true })
  } catch {
    // A locked diagnostic file must not hide the browser results.
  }
}
