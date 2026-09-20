import type { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

import { rememberAttachment } from './image-previews'
import { $backworkspacePage, backworkspaceOwnerKey, type BackworkspaceRoute, editBackworkspacePage } from './page'
import { appendLink, attachmentName, base64FromBytes, imageMarkdown, storableImages } from './paste-image'

interface AttachResult {
  href: string
  path: string
}

// One picture at a time, in the order they were pasted: two round-trips racing
// each other would put the second one's link first. The page's own saves are
// queued the same way (page.ts).
let attachQueue: Promise<void> = Promise.resolve()

/**
 * Pasting a picture into the page stores it beside the page file and writes a
 * relative link where the caret is. Relative, so the page and its images stay
 * one folder — and the agent reading the page can open them.
 */
export function usePasteImage(): { extension: Extension; notice: null | string } {
  const { t } = useI18n()
  const profile = useStore($activeGatewayProfile)
  const connectionId = useStore($activeConnectionId)
  const [failed, setFailed] = useState(false)
  const route = useRef({ connectionId, profile })

  route.current = { connectionId, profile }

  // The link belongs to the page, not to the editor: turning the window back
  // destroys the view while a picture is still being stored, and the paste has
  // to land either way — the same split the agent's reply makes (use-ask-agent).
  const write = useCallback((view: EditorView, link: string, owner: BackworkspaceRoute) => {
    if (!view.dom.isConnected) {
      const page = $backworkspacePage.get()

      // Only into the page it was pasted on. The window may have been turned
      // back and another profile opened by now, and that profile's page has
      // nothing to do with a picture stored in this one's folder.
      if (page?.status === 'ready' && page.key === backworkspaceOwnerKey(owner)) {
        editBackworkspacePage(appendLink(page.content, link))
      }

      return
    }

    // The caret is read now, not when the paste started: storing the picture
    // took a moment and the user may have kept typing. Inserted, never
    // replacing — a selection made since the paste is not the paste's to eat.
    const at = view.state.selection.main.head

    view.dispatch({ changes: { from: at, insert: link }, selection: { anchor: at + link.length } })
  }, [])

  const store = useCallback(
    async (image: Blob, view: EditorView, owner: BackworkspaceRoute) => {
      let attached: AttachResult

      try {
        attached = await requestGatewayForAgent<AttachResult>(
          owner.connectionId,
          owner.profile,
          'backworkspace.attach',
          { data: base64FromBytes(new Uint8Array(await image.arrayBuffer())), name: attachmentName(image) },
          undefined,
          undefined,
          { spawnPriority: 'foreground' }
        )
      } catch {
        setFailed(true)

        return
      }

      // Only the store can fail here. Writing the link is this app's own doing,
      // and dressing a fault of ours up as "that image could not be added"
      // would hide it behind a message about the picture.
      rememberAttachment(attached.href, attached.path)
      write(view, imageMarkdown(attached.href), owner)
    },
    [write]
  )

  const extension = useMemo(
    () =>
      EditorView.domEventHandlers({
        paste(event, view) {
          const images = storableImages(event.clipboardData)

          if (!images.length) {
            return false
          }

          // Whose page this is, read now: the queue may not reach these
          // pictures until the user has moved on to another profile.
          const owner = route.current

          setFailed(false)

          for (const image of images) {
            attachQueue = attachQueue.then(() => store(image, view, owner))
          }

          // A rich copy carries its own text beside the picture, and that text
          // is what the user meant to paste; letting the editor have the event
          // writes it as usual, and the link follows when the picture lands.
          return !event.clipboardData?.getData('text/plain').trim()
        }
      }),
    [store]
  )

  return { extension, notice: failed ? t.backworkspace.attachFailed : null }
}
