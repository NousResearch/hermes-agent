import fs from 'node:fs'

import { CROSS_ENGINE_HERMES_HOME } from './dashboard-cross-engine-home'

export default async function globalSetup(): Promise<void> {
  // The PID-qualified directory is unique to this Playwright run. Avoid a
  // recursive delete at startup: Windows Defender/ACLs can briefly retain a
  // prior temp directory even after its browser process has exited.
  fs.mkdirSync(CROSS_ENGINE_HERMES_HOME, { recursive: true })
  fs.writeFileSync(
    `${CROSS_ENGINE_HERMES_HOME}/config.yaml`,
    '# Cross-engine browser smoke: provider-free isolated dashboard\n',
    'utf8',
  )
}
