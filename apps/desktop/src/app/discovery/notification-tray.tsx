import './discovery.css'

import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { $renameSplashReopenable, reopenRenameSplash } from '@/store/rename-splash'

import { useRailMenuCollisionPadding } from '../redesign/portal-status-menu'

import { useDiscoveryCopy } from './discovery-copy'
import { notificationTray } from './tray-items'

const NOTHING_UNSEEN: ReadonlySet<string> = new Set()

/**
 * The rail's bell and its tray (Figma E6 18677:29931, as a popover): what the app has to tell you,
 * one row per item of the registry (tray-items.tsx) in one card, then the introduction's way back
 * and Mark all read. A dot on the bell while an item has news you have not seen; opening the tray
 * sees it, and the rows that were new say so until it closes or Mark all read clears them.
 */
export function NotificationBell() {
  const copy = useDiscoveryCopy()
  const { pathname, search } = useLocation()
  const listed = useStore(notificationTray.$listed)
  const unseen = useStore(notificationTray.$unseen)
  const introduction = useStore($renameSplashReopenable)
  const [open, setOpen] = useState(false)
  // What was new when the tray opened: seeing it clears the bell, not the cards' marks.
  const [newOnOpen, setNewOnOpen] = useState(NOTHING_UNSEEN)
  const collisionPadding = useRailMenuCollisionPadding()
  useEffect(() => setOpen(false), [pathname, search])
  // A card that takes the person elsewhere (the splash, an update) keeps focus there on close.
  const handedOff = useRef(false)

  const handOff = () => {
    handedOff.current = true
    setOpen(false)
  }

  const onOpenChange = (next: boolean) => {
    if (next) {
      setNewOnOpen(notificationTray.$unseen.get())
      notificationTray.markAllSeen()
    }

    setOpen(next)
  }

  const hasNews = unseen.size > 0

  return (
    <Popover onOpenChange={onOpenChange} open={open}>
      <Tip label={copy.tray.title} placement="left-rail">
        <PopoverTrigger asChild>
          <Button
            aria-description={hasNews ? copy.tray.unseen : undefined}
            aria-label={copy.tray.title}
            className="portal-chrome-button"
            data-nav-spot="notifications"
            size="icon"
            variant="ghost"
          >
            <Codicon name="bell" />
            {hasNews && <span aria-hidden data-slot="rail-alert" data-tone="news" />}
          </Button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent
        align="end"
        aria-label={copy.tray.title}
        className="portal-rail-popover discovery-tray"
        collisionPadding={collisionPadding}
        data-slot="notification-tray"
        onCloseAutoFocus={event => {
          if (handedOff.current) {
            handedOff.current = false
            event.preventDefault()
          }
        }}
        side="right"
        sideOffset={0}
        variant="menu"
      >
        <h2 className="discovery-tray-title">{copy.tray.heading}</h2>
        {listed.length === 0 ? (
          <p className="discovery-tray-empty">{copy.tray.empty}</p>
        ) : (
          <>
            <ul className="discovery-welcome-card discovery-tray-items">
              {listed.map(({ Card, id }) => (
                <li key={id}>
                  <Card close={handOff} dismiss={() => setOpen(false)} unseen={newOnOpen.has(id)} />
                </li>
              ))}
            </ul>
            {(introduction || newOnOpen.size > 0) && (
              <div className="discovery-welcome-actions">
                {introduction && (
                  <Button
                    className="discovery-welcome-action"
                    onClick={() => {
                      handOff()
                      reopenRenameSplash()
                    }}
                  >
                    {copy.tray.openIntroduction}
                  </Button>
                )}
                {newOnOpen.size > 0 && (
                  <Button
                    className="discovery-welcome-action"
                    onClick={() => setNewOnOpen(NOTHING_UNSEEN)}
                    variant="secondary"
                  >
                    {copy.tray.markAllRead}
                  </Button>
                )}
              </div>
            )}
            <p className="discovery-welcome-footnote">{copy.tray.footnote}</p>
          </>
        )}
      </PopoverContent>
    </Popover>
  )
}
