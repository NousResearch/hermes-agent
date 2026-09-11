/** Machine facts load before the runbook so a new computer or Spark can lead
 *  with machine setup as the first task. */

import { atom } from 'nanostores'

import type { DesktopMachineProfile } from '@/global'

/** Allow time to get around to setup without treating a daily-use machine as new. */
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

/** Unknown counts as not-new: the option is always offered, it just doesn't
 *  lead unless we can see a reason for it to. */
export function machineLooksNew(): boolean {
  const age = $machine.get()?.ageDays

  return age != null && age <= NEW_MACHINE_DAYS
}

/** Login names that are not a name. 'akp' suggests fine; 'user' does not. */
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

/** Name the OS language for the model, independent of the UI's bundled locales.
 *  English, missing or invalid tags need no language instruction. */
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

/** True when setting the machine up should be the only thing on offer, with
 *  everything else folded away behind one more tap. */
export function machineSetupLeads(): boolean {
  return machineIsSpark() || machineLooksNew()
}

/** What the user calls the thing in front of them. */
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

/** Age leads the setup brief because a new machine needs work that a
 *  daily-use machine may already have done. */
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
