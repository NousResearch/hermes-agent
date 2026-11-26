import '@/styles.css'

import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { ProfileRail } from '@/app/chat/sidebar/profile-switcher'
import { AppContextMenu } from '@/app/context-menu/app-context-menu'
import { BootFailureModal } from '@/components/boot-failure-modal'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { RootTooltipProvider, Tip } from '@/components/ui/tooltip'
import { createBrowserZoom } from '@/lib/browser-zoom'
import { $activeGatewayProfile, $profiles } from '@/store/profile'

const profileNames = [
  'default',
  'alpha',
  'bravo',
  'charlie',
  'delta',
  'echo',
  'foxtrot',
  'golf',
  'hotel',
  'india',
  'juliet',
  'kilo',
  'lima',
  'mike',
  'november',
  'oscar',
  'papa',
  'quebec',
  'romeo',
  'sierra',
  'tango',
  'zulu'
]

const seededProfiles = profileNames.map(name => ({
  exists: true,
  has_env: false,
  is_active: name === 'alpha',
  is_default: name === 'default',
  model: null,
  name,
  path: `profiles/${name}`,
  provider: null,
  skill_count: 0
}))

$profiles.set(seededProfiles)
$activeGatewayProfile.set('alpha')

const zoom = createBrowserZoom('/webapp-scaled-poppers-test')
Object.assign(window, { setTestZoom: zoom.setPercent })

function Fixture() {
  const [dialogOpen, setDialogOpen] = useState(false)
  const [bootModalOpen, setBootModalOpen] = useState(false)
  const [sheetOpen, setSheetOpen] = useState(false)

  return (
    <MemoryRouter>
      <RootTooltipProvider>
        <div
          className="flex h-full w-full flex-col bg-(--ui-chat-surface-background) text-foreground"
          data-contrib-shell=""
        >
          <header className="flex shrink-0 items-center gap-3 border-b border-(--ui-stroke-tertiary) px-3 py-2">
            <Select defaultValue="opt-1">
              <SelectTrigger className="w-48 shrink-0" data-testid="body-select">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: 24 }, (_, index) => (
                  <SelectItem key={index} value={`opt-${index + 1}`}>
                    Option {index + 1}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Popover>
              <PopoverTrigger asChild>
                <button
                  className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
                  data-testid="body-popover"
                  type="button"
                >
                  Open Popover
                </button>
              </PopoverTrigger>
              <PopoverContent align="center" side="bottom">
                <button
                  className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
                  data-testid="popover-action"
                  onClick={() => {
                    window.document.body.dataset.popoverClicked = 'true'
                  }}
                  type="button"
                >
                  Confirm popover
                </button>
              </PopoverContent>
            </Popover>

            <Tip delayDuration={0} label="Header tooltip label" side="bottom">
              <button
                className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
                data-testid="body-tip"
                type="button"
              >
                Hover Tip
              </button>
            </Tip>

            <button
              className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
              data-testid="open-dialog"
              onClick={() => setDialogOpen(true)}
              type="button"
            >
              Open Dialog
            </button>

            <button
              className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
              data-testid="open-boot-modal"
              onClick={() => setBootModalOpen(true)}
              type="button"
            >
              Open BootModal
            </button>

            <button
              className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
              data-testid="open-sheet"
              onClick={() => setSheetOpen(true)}
              type="button"
            >
              Open Sheet
            </button>
          </header>

          <div className="flex min-h-0 flex-1">
            <aside
              className="flex w-60 shrink-0 flex-col border-r border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background)"
              data-slot="sidebar-container"
            >
              <div className="flex-1" />
              <div className="shrink-0 px-0.5 pt-0.5 pb-1">
                <ProfileRail />
              </div>
            </aside>

            <main className="flex min-w-0 flex-1 flex-col justify-between p-4">
              <div className="flex-1" data-testid="context-zone">
                <a data-testid="context-link" href="https://example.com">
                  Right-click link
                </a>
              </div>

              <div className="flex items-center justify-end gap-3">
                <Tip delayDuration={0} label="Composer tooltip label" side="top">
                  <button
                    className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
                    data-testid="br-tip"
                    type="button"
                  >
                    BR Tip
                  </button>
                </Tip>

                <Popover>
                  <PopoverTrigger asChild>
                    <button
                      className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
                      data-testid="br-popover"
                      type="button"
                    >
                      BR Popover
                    </button>
                  </PopoverTrigger>
                  <PopoverContent align="end" side="top">
                    <div>Bottom right popover</div>
                  </PopoverContent>
                </Popover>

                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      className="rounded border border-(--ui-stroke-primary) px-2.5 py-1 text-xs"
                      data-testid="model-pill"
                      type="button"
                    >
                      Model picker
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-72" side="top" sideOffset={8}>
                    {Array.from({ length: 20 }, (_, index) => (
                      <DropdownMenuItem key={index}>Model {index + 1}</DropdownMenuItem>
                    ))}
                    <DropdownMenuSub>
                      <DropdownMenuSubTrigger data-testid="model-sub-trigger">More models</DropdownMenuSubTrigger>
                      <DropdownMenuSubContent>
                        {Array.from({ length: 10 }, (_, index) => (
                          <DropdownMenuItem
                            key={index}
                            onSelect={() => {
                              window.document.body.dataset.submodelPicked = `Submodel ${index + 1}`
                            }}
                          >
                            Submodel {index + 1}
                          </DropdownMenuItem>
                        ))}
                      </DropdownMenuSubContent>
                    </DropdownMenuSub>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </main>
          </div>

          <footer className="flex h-7 shrink-0 items-center justify-between border-t border-(--ui-stroke-tertiary) px-3 text-xs text-muted-foreground">
            <span>Backend ready</span>
          </footer>

          <AppContextMenu />

          <Dialog onOpenChange={setDialogOpen} open={dialogOpen}>
            <DialogContent>
              <DialogTitle>Dialog Poppers</DialogTitle>
              <div className="flex items-center gap-3 py-2">
                <Select defaultValue="d-1">
                  <SelectTrigger className="w-48" data-testid="dialog-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: 20 }, (_, index) => (
                      <SelectItem key={index} value={`d-${index + 1}`}>
                        Dialog Option {index + 1}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Popover>
                  <PopoverTrigger asChild>
                    <button
                      className="rounded border border-(--ui-stroke-secondary) px-2 py-1 text-xs"
                      data-testid="dialog-popover"
                      type="button"
                    >
                      Dialog Popover
                    </button>
                  </PopoverTrigger>
                  <PopoverContent side="bottom">
                    <div>Inside dialog</div>
                  </PopoverContent>
                </Popover>
              </div>
            </DialogContent>
          </Dialog>

          {bootModalOpen ? (
            <BootFailureModal onDismiss={() => setBootModalOpen(false)} title="Boot failure">
              <div className="rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) p-6">
                <Select defaultValue="b-1">
                  <SelectTrigger className="w-48" data-testid="boot-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: 20 }, (_, index) => (
                      <SelectItem key={index} value={`b-${index + 1}`}>
                        Boot Option {index + 1}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </BootFailureModal>
          ) : null}

          <Sheet onOpenChange={setSheetOpen} open={sheetOpen}>
            <SheetContent className="w-64 p-2" side="left">
              <SheetTitle>Mobile Sidebar</SheetTitle>
              <div className="flex-1" />
              <div data-testid="sheet-profile-rail">
                <ProfileRail />
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </RootTooltipProvider>
    </MemoryRouter>
  )
}

createRoot(window.document.getElementById('root')!).render(<Fixture />)
