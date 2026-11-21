/**
 * Telling that a newer Webapp build is deployed, and when a tab may pick it up.
 *
 * The host serves `index.html` with `Cache-Control: no-store` and the bundle under content-hashed
 * names (`assets/index-<hash>.js`, immutable), so the entry module an index names IS the build:
 * a rebuild with any change gets a new entry name. A tab compares the entry it booted from with the
 * one the host serves now. Nothing else is compared, so a host that injects per-request values into
 * the page (its token, base path, theme) never reads as a new build.
 */

/** The first module script an HTML document loads, as written in its `src` attribute. */
export function entryModuleOf(html: string): null | string {
  for (const tag of html.match(/<script\b[^>]*>/gi) ?? []) {
    if (!/\btype\s*=\s*["']?module\b/i.test(tag)) {
      continue
    }

    const src = /\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(tag)
    const value = src?.[1] ?? src?.[2] ?? src?.[3]

    if (value) {
      return value
    }
  }

  return null
}

const directoryOf = (src: string) => src.slice(0, src.lastIndexOf('/') + 1)

/** A different entry in the same place: a new build of this app. Anything else (no entry, a page
 *  from somewhere else such as a sign-in wall) proves nothing. */
export function isNewerBuild(running: null | string, served: null | string): boolean {
  return Boolean(running && served && served !== running && directoryOf(served) === directoryOf(running))
}

export interface ApplyConditions {
  /** Chats with a turn running in this tab. */
  workingChats: number
  /** Live speech is playing a reply. */
  voiceActive: boolean
}

/**
 * Whether opening a new chat may load the new build. A reload there loses nothing on screen (the
 * new chat is empty), but it would cut the live view of a turn running in another chat or a reply
 * being spoken, so it waits for those to finish: the next new chat after them applies it. The same
 * work keeps a desktop window from quitting (store/active-work.ts).
 */
export function mayApplyOnNewChat(conditions: ApplyConditions): boolean {
  return conditions.workingChats === 0 && !conditions.voiceActive
}
