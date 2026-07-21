import { type ComponentType, lazy } from 'react'

export interface ProviderConfigSurfaceProps {
  provider: string
}

const LazyOpenVikingConfigPanel = lazy(async () => {
  const { OpenVikingConfigPanel } = await import('./openviking-config-panel')

  return { default: OpenVikingConfigPanel }
})

function OpenVikingConfigSurface() {
  return <LazyOpenVikingConfigPanel />
}

const PROVIDER_CONFIG_SURFACES: Record<string, ComponentType<ProviderConfigSurfaceProps>> = {
  openviking: OpenVikingConfigSurface
}

export function getProviderConfigSurface(
  surface: null | string | undefined
): ComponentType<ProviderConfigSurfaceProps> | null {
  return surface ? (PROVIDER_CONFIG_SURFACES[surface] ?? null) : null
}
