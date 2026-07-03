import type * as React from 'react'

import type { DashboardPluginRegistryListener } from './types'

const registered = new Map<string, React.ComponentType>()
const loadErrors = new Map<string, string>()
const listeners = new Set<DashboardPluginRegistryListener>()

function notifyRegistry() {
  for (const listener of listeners) {
    try {
      listener()
    } catch {
      // Registry subscribers should not be able to break plugin loading.
    }
  }
}

export function registerDashboardPlugin(name: string, component: React.ComponentType): void {
  loadErrors.delete(name)
  registered.set(name, component)
  notifyRegistry()
}

export function getDashboardPluginComponent(name: string): React.ComponentType | undefined {
  return registered.get(name)
}

export function getDashboardPluginLoadError(name: string): string | undefined {
  return loadErrors.get(name)
}

export function setDashboardPluginLoadError(name: string, message: string): void {
  loadErrors.set(name, message)
  notifyRegistry()
}

export function notifyDashboardPluginRegistry(): void {
  notifyRegistry()
}

export function onDashboardPluginRegistered(listener: DashboardPluginRegistryListener): () => void {
  listeners.add(listener)

  return () => listeners.delete(listener)
}

export function resetDashboardPluginRegistryForTests(): void {
  registered.clear()
  loadErrors.clear()
  notifyRegistry()
}
