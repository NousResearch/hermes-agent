import os from 'node:os'
import path from 'node:path'

export const CROSS_ENGINE_HERMES_HOME = path.join(
  os.tmpdir(),
  `hermes-dashboard-cross-engine-${process.pid}`,
)

export const CROSS_ENGINE_PORT = Number(process.env.HERMES_CROSS_ENGINE_PORT ?? '19119')
