/** Machine facts load before the runbook is built so a new computer or a Spark can lead with machine setup as
 *  the first task. */

import { atom } from 'nanostores'

import type { DesktopMachineProfile } from '@/global'

/** 21 days leaves time to finish setup without counting a daily-use machine as new. */
const NEW_MACHINE_DAYS = 21

export const $machine = atom<DesktopMachineProfile | null>(null)

export async function loadMachineProfile(): Promise<void> {
  if ($machine.get()) {
    return
  }

  const profile = await window.hermesDesktop?.getMachineProfile?.().catch(() => null)

  if (profile) {
    $machine.set(profile)
  }
}

/** An unknown age counts as not new. Machine setup is still offered; age makes it lead only when the age is
 *  known and within NEW_MACHINE_DAYS. */
export function machineLooksNew(): boolean {
  const age = $machine.get()?.ageDays

  return age != null && age <= NEW_MACHINE_DAYS
}

/** Generic login names that are not a person's name. A short handle such as 'akp' is still usable. */
const NON_NAME_USERNAMES = new Set([
  'admin',
  'administrator',
  'default',
  'guest',
  'me',
  'owner',
  'root',
  'test',
  'user'
])

/** The login handle is only a suggestion; the user still chooses their name.
 *  Null means the guide asks without a default. */
export function machineUserName(): string | null {
  const raw = ($machine.get()?.username ?? '').trim()

  if (raw.length < 2 || raw.length > 20) {
    return null
  }

  return NON_NAME_USERNAMES.has(raw.toLowerCase()) ? null : raw
}

/** Names the OS language for the model, independent of the UI's bundled locales. Returns null for English and
 *  for a missing or invalid tag, which need no language instruction. */
export function machineLanguageName(): string | null {
  const tag = ($machine.get()?.locale ?? '').trim()

  if (!tag || /^en\b/i.test(tag)) {
    return null
  }

  try {
    const name = new Intl.DisplayNames(['en'], { fallback: 'code', type: 'language' }).of(tag)

    return name && name.toLowerCase() !== 'english' ? name : null
  } catch {
    return null
  }
}

/** Recognize RTX Sparks by platform/architecture/GPU and DGX Sparks by model.
 *  Device-tree underscores separate words: real units report NVIDIA_DGX_Spark. */
export function machineIsSpark(): boolean {
  const profile = $machine.get()

  if (!profile) {
    return false
  }

  const rtx = profile.platform === 'win32' && profile.arch === 'arm64' && profile.nvidia
  const dgx = /\b(dgx|spark|gb10)\b/i.test(profile.model.replace(/_/g, ' '))

  return rtx || dgx
}

/** True when machine setup should be the only first task shown, with the other options behind one more tap. */
export function machineSetupLeads(): boolean {
  return machineIsSpark() || machineLooksNew()
}

export function machineKind(): string {
  if (machineIsSpark()) {
    return 'Spark'
  }

  switch ($machine.get()?.platform) {
    case 'darwin':
      return 'Mac'

    case 'win32':
      return 'PC'

    default:
      return 'computer'
  }
}

/** The age comes first in the description because a new machine needs setup work that a daily-use machine may
 *  already have done. */
export function machineDescription(): string {
  const profile = $machine.get()

  if (!profile) {
    return ''
  }

  return [
    machineLooksNew() ? `set up ${daysAgo(profile.ageDays)}` : '',
    machineIsSpark() ? 'an NVIDIA Spark' : profile.nvidia ? 'has an NVIDIA GPU' : '',
    profile.model,
    `${profile.platform} ${profile.release}`,
    profile.arch
  ]
    .filter(Boolean)
    .join(', ')
}

function daysAgo(days: null | number): string {
  if (days === 0) {
    return 'today'
  }

  return days === 1 ? 'yesterday' : `${days} days ago`
}
