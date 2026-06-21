import type { Settings } from './core/types';

export const DEFAULT_SETTINGS: Settings = { provider: 'openai', defaultTier: 'take', language: '', disclaimerAccepted: false };

export async function getSettings(): Promise<Settings> {
  const stored = await chrome.storage.local.get('settings');
  return { ...DEFAULT_SETTINGS, ...(stored.settings ?? {}) };
}

export async function saveSettings(settings: Settings): Promise<void> {
  await chrome.storage.local.set({ settings });
}
