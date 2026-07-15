import { useCallback, useEffect, useState } from 'react'

import { downloadGatewayMediaFile, isRemoteGateway, mediaExternalUrl } from '@/lib/media'

export function useOpenMediaFile(path?: string) {
  const [openFailed, setOpenFailed] = useState(false)
  const source = path?.trim()
  const downloadsRemoteFile = Boolean(source && !/^https?:/i.test(source) && window.hermesDesktop && isRemoteGateway())

  useEffect(() => setOpenFailed(false), [source])

  const open = useCallback(() => {
    if (!source) {
      return
    }

    setOpenFailed(false)

    const protocol = source.match(/^([a-z][a-z0-9+.-]*):/i)?.[1]?.toLowerCase()
    const absoluteLocalPath = /^(?:\/|[a-z]:[\\/]|\\\\)/i.test(source)

    if ((protocol && !['file', 'http', 'https'].includes(protocol)) || (!protocol && !absoluteLocalPath)) {
      setOpenFailed(true)

      return
    }

    if (downloadsRemoteFile) {
      void downloadGatewayMediaFile(source).catch(() => setOpenFailed(true))

      return
    }

    const openExternal = window.hermesDesktop?.openExternal

    if (!openExternal) {
      setOpenFailed(true)

      return
    }

    void openExternal(mediaExternalUrl(source)).catch(() => setOpenFailed(true))
  }, [downloadsRemoteFile, source])

  return { downloadsRemoteFile, open, openFailed }
}
