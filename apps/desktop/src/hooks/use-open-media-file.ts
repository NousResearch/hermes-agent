import { useCallback, useEffect, useRef, useState } from 'react'

import { downloadGatewayMediaFile, isRemoteGateway, mediaExternalUrl } from '@/lib/media'

export function useOpenMediaFile(path?: string) {
  const [openFailed, setOpenFailed] = useState(false)
  const source = path?.trim()
  const previousSource = useRef(source)
  const requestGeneration = useRef(0)

  if (previousSource.current !== source) {
    previousSource.current = source
    requestGeneration.current += 1
  }

  const downloadsRemoteFile = Boolean(
    source &&
    window.hermesDesktop &&
    isRemoteGateway() &&
    (/^(?:\/|[a-z]:[\\/]|\\\\)/i.test(source) || !/^[a-z][a-z0-9+.-]*:/i.test(source) || /^file:/i.test(source))
  )

  useEffect(() => setOpenFailed(false), [source])

  const open = useCallback(() => {
    if (!source) {
      return
    }

    setOpenFailed(false)

    const generation = ++requestGeneration.current

    const failCurrentRequest = () => {
      if (requestGeneration.current === generation) {
        setOpenFailed(true)
      }
    }

    const protocol = source.match(/^([a-z][a-z0-9+.-]*):/i)?.[1]?.toLowerCase()
    const absoluteLocalPath = /^(?:\/|[a-z]:[\\/]|\\\\)/i.test(source)

    if (downloadsRemoteFile) {
      void downloadGatewayMediaFile(source).catch(failCurrentRequest)

      return
    }

    if (
      (protocol && !absoluteLocalPath && !['file', 'http', 'https'].includes(protocol)) ||
      (!protocol && !absoluteLocalPath)
    ) {
      failCurrentRequest()

      return
    }

    const openExternal = window.hermesDesktop?.openExternal

    if (!openExternal) {
      failCurrentRequest()

      return
    }

    void openExternal(mediaExternalUrl(source)).catch(failCurrentRequest)
  }, [downloadsRemoteFile, source])

  return { downloadsRemoteFile, open, openFailed }
}
