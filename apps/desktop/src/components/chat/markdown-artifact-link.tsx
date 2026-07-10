import { useStore } from '@nanostores/react'
import type { ComponentProps } from 'react'
import { useEffect, useRef, useState } from 'react'

import { translateNow } from '@/i18n'
import { desktopFilesystemKey, desktopFsCacheKey } from '@/lib/desktop-fs'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { markdownArtifactFileTarget } from '@/lib/preview-targets'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { setCurrentSessionPreviewTarget } from '@/store/preview'
import { $currentCwd } from '@/store/session'

interface MarkdownArtifactLinkProps extends Omit<ComponentProps<'a'>, 'href'> {
  href: string
}

export function MarkdownArtifactLink({ children, className, href, ...props }: MarkdownArtifactLinkProps) {
  const cwd = useStore($currentCwd)
  const [opening, setOpening] = useState(false)
  const mountedRef = useRef(false)
  const requestRef = useRef(0)

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
      requestRef.current += 1
    }
  }, [])

  const open = async () => {
    if (opening) {
      return
    }

    const request = ++requestRef.current
    const connectionKey = desktopFsCacheKey()
    setOpening(true)

    try {
      const fileTarget = markdownArtifactFileTarget(href)
      const target = fileTarget ? await normalizeOrLocalPreviewTarget(fileTarget, cwd || undefined) : null

      if (!mountedRef.current || request !== requestRef.current || connectionKey !== desktopFsCacheKey()) {
        return
      }

      if (!target || target.kind !== 'file') {
        throw new Error(`Could not open Markdown artifact: ${href}`)
      }

      setCurrentSessionPreviewTarget(
        { ...target, artifact: true, filesystemKey: desktopFilesystemKey() },
        'artifact-link',
        href
      )
    } catch (error) {
      if (mountedRef.current && request === requestRef.current) {
        notifyError(error, translateNow('preview.unavailable'))
      }
    } finally {
      if (mountedRef.current && request === requestRef.current) {
        setOpening(false)
      }
    }
  }

  return (
    <a
      {...props}
      aria-busy={opening || undefined}
      className={cn(
        'font-semibold text-foreground underline underline-offset-4 decoration-current/20 wrap-anywhere',
        className
      )}
      href={href}
      onClick={event => {
        event.preventDefault()
        void open()
      }}
    >
      {children}
    </a>
  )
}
