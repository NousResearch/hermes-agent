/**
 * What Hermes knows about the computer it was just installed on.
 *
 * One question, really: is this machine new? A brand-new computer makes "set
 * this thing up for me" the likeliest first task rather than one option among
 * several — drivers, updates, a package manager, the tools they just told us
 * they use — and it is a task Hermes can do end to end with no account
 * anywhere, which is exactly what the first build has to be.
 *
 * A Spark answers yes on its own, without waiting for the account age to say
 * so; see machineIsSpark.
 *
 * Loaded once, before the guided chat's runbook is composed.
 */

import { atom } from 'nanostores'

import type { DesktopMachineProfile } from '@/global'

/** A computer this young is almost certainly still being set up. Wide enough
 *  to cover the week someone spends getting around to it, short enough that a
 *  machine in daily use never trips it. */
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

/** A suggestable name for the guided chat's first question: the OS account
 *  name, when it actually looks like something you could be called. The login
 *  handle is a hint, never a truth — the greeting offers it as a default and
 *  the user still picks. Null means no suggestion; the guide just asks. */
export function machineUserName(): string | null {
  const raw = ($machine.get()?.username ?? '').trim()

  if (raw.length < 2 || raw.length > 20) {
    return null
  }

  return NON_NAME_USERNAMES.has(raw.toLowerCase()) ? null : raw
}

/** The OS display language, as an English language name the model can act on
 *  ("Japanese", "Brazilian Portuguese"). Null when the machine hasn't
 *  answered, or when it is already English.
 *
 *  Deliberately NOT limited to the six locales the UI ships: the chrome can
 *  only be translated where a bundle exists, but the MODEL speaks whatever the
 *  user does, and a Portuguese speaker being answered in Portuguese matters
 *  more than the sidebar labels matching. `Intl.DisplayNames` turns the raw
 *  tag into the name of a language rather than a code the model has to
 *  decode — and its own fallback ('code') hands back the tag if it doesn't
 *  know it either, which is still a better instruction than nothing. */
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

/** A Spark, either kind. Neither is a machine anyone owns for its own sake —
 *  both are bought to be set up — so one takes the front of the flow whatever
 *  its account age says.
 *
 *  The two are identified differently because they are different computers.
 *  An **RTX Spark** is a Windows-on-Arm PC (the N1X superchip, in this fall's
 *  ASUS / Dell / HP / Lenovo / Surface / MSI laptops and mini desktops); the
 *  OEM badge on the case is not a name we can enumerate, so it is recognised
 *  by its shape — Windows, Arm, NVIDIA silicon, a combination nothing else
 *  currently ships. A **DGX Spark** is the Linux GB10 developer box, and it
 *  says so in the device tree.
 *
 *  On that string, underscores are separators rather than letters: a real unit
 *  reports `NVIDIA_DGX_Spark`, which \b reads as ONE word and would never
 *  match. */
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

/** One line for the machine-setup brief, so the agent that picks the job up
 *  starts knowing what it is looking at instead of asking.
 *
 *  Age leads, because it is the fact that changes the work: on a machine
 *  someone unboxed this week the drivers, updates and toolchain are genuinely
 *  undone, and doing them is worth an afternoon of the user's life. On a
 *  machine that has been running for two years most of it is already handled,
 *  and an agent that doesn't know that will "fix" things that were never
 *  broken. */
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

export function resetMachineProfileForTests(): void {
  $machine.set(null)
}
