import type { CatalogPluginPresence, MachineFactsResult } from '@hermes/shared'

import type { Answers, Facts } from './flow'

type AppState = CatalogPluginPresence['state']

const TITLES = new Map([
  ['blender', 'Blender'],
  ['nvidia-app', 'NVIDIA App'],
  ['nvidia-broadcast', 'NVIDIA Broadcast']
])

export function presence(apps: Record<string, AppState>): CatalogPluginPresence[] {
  return Object.entries(apps).map(([name, state]) => ({
    disclosure: `${TITLES.get(name)} disclosure.`,
    name,
    sentence: '',
    state,
    title: TITLES.get(name) ?? name
  }))
}

function machine(overrides: Partial<MachineFactsResult> & { info: Partial<MachineFactsResult['machine']> }) {
  const { info, ...rest } = overrides

  return {
    full_name: 'Sid Example',
    has_nvidia_gpu: false,
    is_spark: false,
    machine: info,
    machine_kind: 'PC',
    ...rest
  } satisfies MachineFactsResult
}

const CONNECTORS: Facts['connectors'] = {
  rows: [
    { id: 'github', label: 'GitHub' },
    { id: 'gmail', label: 'Gmail' },
    { id: 'slack', label: 'Slack' }
  ],
  status: 'ready'
}

/** Four machines, presence as the backend answers it (wrong-OS names already dropped). */
export const FIXTURES = {
  spark: {
    connectors: CONNECTORS,
    local: { id: 'qwen3.8-27b', name: 'Qwen3.8 27B' },
    machine: machine({
      has_nvidia_gpu: true,
      info: { cpu_model: 'NVIDIA N1X', gpu_class: 'nvidia', native_arch: 'arm64', os_family: 'win32', os_release: '11', ram_gb: 128 },
      is_spark: true,
      machine_kind: 'Spark'
    }),
    plugins: presence({ blender: 'present', 'nvidia-app': 'present', 'nvidia-broadcast': 'missing_app' })
  },
  windowsRtx: {
    connectors: CONNECTORS,
    local: { id: 'qwen3.6-35b-a3b', name: 'Qwen3.6 35B-A3B' },
    machine: machine({
      has_nvidia_gpu: true,
      info: { cpu_model: 'AMD Ryzen 7 7800X3D', gpu_class: 'nvidia', os_family: 'win32', os_release: '11', ram_gb: 32 }
    }),
    plugins: presence({ blender: 'missing_app', 'nvidia-app': 'present', 'nvidia-broadcast': 'app_not_running' })
  },
  windowsLaptop: {
    connectors: CONNECTORS,
    local: null,
    machine: machine({ info: { cpu_model: 'Intel Core Ultra 7', os_family: 'win32', os_release: '11', ram_gb: 16 } }),
    plugins: presence({ blender: 'missing_app', 'nvidia-app': 'missing_app', 'nvidia-broadcast': 'unknown' })
  },
  mac: {
    connectors: CONNECTORS,
    local: { id: 'qwen3.6-35b-a3b', name: 'Qwen3.6 35B-A3B' },
    machine: machine({
      info: { cpu_model: 'Apple M4 Pro', os_family: 'darwin', os_release: '26', ram_gb: 48 },
      machine_kind: 'Mac'
    }),
    plugins: presence({ blender: 'present' })
  }
} satisfies Record<string, Facts>

export const answers = (patch: Partial<Answers> = {}): Answers => ({ apps: [], connectors: [], skipped: [], ...patch })
