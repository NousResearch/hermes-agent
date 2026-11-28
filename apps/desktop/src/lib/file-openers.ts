import type { Contribution } from '@/contrib/types'

export const FILE_OPENER_AREA = 'preview.fileOpener'

export interface FileOpenerContribution {
  claims: (path: string) => boolean
  open: (path: string) => void | Promise<void>
}

export function tryOpenFileWithPlugin(
  path: string,
  contributions: readonly Contribution[],
  fallback: () => void,
  logger: Pick<Console, 'warn'> = console
): void {
  for (const contribution of contributions) {
    const opener = contribution.data as Partial<FileOpenerContribution> | undefined

    if (typeof opener?.claims !== 'function' || typeof opener.open !== 'function') {
      continue
    }

    let claims: boolean

    try {
      claims = opener.claims(path)
    } catch (error) {
      logger.warn(`File opener ${contribution.id} failed to check ${path}; using the next opener`, error)

      continue
    }

    if (!claims) {
      continue
    }

    try {
      void Promise.resolve(opener.open(path)).catch(error => {
        logger.warn(`File opener ${contribution.id} failed to open ${path}; using the built-in preview`, error)
        fallback()
      })
    } catch (error) {
      logger.warn(`File opener ${contribution.id} failed to open ${path}; using the built-in preview`, error)
      fallback()
    }

    return
  }

  fallback()
}
