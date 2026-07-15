import { useCallback, useEffect, useRef, useState } from 'react'

import { downloadGatewayMediaFile, isRemoteGateway, mediaExternalUrl } from '@/lib/media'

export function useOpenMediaFile(path?: string) {
  const [openFailed, setOpenFailed] = useState(false)
  const source = path?.trim()
  const currentSource = useRef(source)

  currentSource.current = source

  const downloadsRemoteFile = Boolean(source && !/^https?:/i.test(source) && window.hermesDesktop && isRemoteGateway())

  useEffect(() => setOpenFailed(false), [source])

  const open = useCallback(() => {
    if (!source) {
      return
    }

    setOpenFailed(false)

    const failCurrentRequest = () => {
      if (currentSource.current === source) {
        setOpenFailed(true)
      }
    }

    const protocol = source.match(/^([a-z][a-z0-9+.-]*):/i)?.[1]?.toLowerCase()
    const absoluteLocalPath = /^(?:\/|[a-z]:[\\/]|\\\\)/i.test(source)

    if (
      (protocol && !absoluteLocalPath && !['file', 'http', 'https'].includes(protocol)) ||
      (!protocol && !absoluteLocalPath)
    ) {
      failCurrentRequest()

      return
    }

    if (downloadsRemoteFile) {
      void downloadGatewayMediaFile(source).catch(failCurrentRequest)

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
