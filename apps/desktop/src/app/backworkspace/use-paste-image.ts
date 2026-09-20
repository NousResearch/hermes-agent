import type { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

import { rememberAttachment } from './image-previews'
import { writeToPage } from './live-editor'
import { backworkspaceOwnerKey, type BackworkspaceRoute } from './page'
import { attachmentName, base64FromBytes, imageMarkdown, linkOnOwnLine, storableImages } from './paste-image'

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

  // The link belongs to the page, not to the editor it was pasted into: the
  // window may be turned while a picture is still being stored, and the paste
  // has to land either way — the same as the agent's reply (use-ask-agent).
  const write = useCallback((link: string, owner: BackworkspaceRoute) => {
    writeToPage(backworkspaceOwnerKey(owner), (doc, caret) =>
      caret === null
        ? { from: doc.length, insert: linkOnOwnLine(doc, link) }
        : // The caret as it is now, not when the paste started: storing the
          // picture took a moment and the user may have kept typing. Inserted,
          // never replacing — a selection made since is not the paste's to eat.
          { caret: caret + link.length, from: caret, insert: link }
    )
  }, [])

  const store = useCallback(
    async (image: Blob, owner: BackworkspaceRoute) => {
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
      write(imageMarkdown(attached.href), owner)
    },
    [write]
  )

  const extension = useMemo(
    () =>
      EditorView.domEventHandlers({
        paste(event) {
          const images = storableImages(event.clipboardData)

          if (!images.length) {
            return false
          }

          // Whose page this is, read now: the queue may not reach these
          // pictures until the user has moved on to another profile.
          const owner = route.current

          setFailed(false)

          for (const image of images) {
            attachQueue = attachQueue.then(() => store(image, owner))
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
