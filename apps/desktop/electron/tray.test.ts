import { beforeEach, describe, expect, it, vi } from 'vitest'

const electronMock = vi.hoisted(() => {
  type ClickHandler = () => void
  type MenuItem = { label?: string; click?: ClickHandler; type?: string }

  const trayInstances: FakeTray[] = []

  class FakeTray {
    private readonly listeners = new Map<string, ClickHandler[]>()
    destroyed = false
    tooltip?: string
    menu?: MenuItem[]

    setToolTip = vi.fn((tooltip: string) => {
      this.tooltip = tooltip
    })

    setContextMenu = vi.fn((menu: { template: MenuItem[] }) => {
      this.menu = menu.template
    })

    on = vi.fn((event: string, handler: ClickHandler) => {
      const list = this.listeners.get(event) ?? []
      list.push(handler)
      this.listeners.set(event, list)
      return this
    })

    destroy = vi.fn(() => {
      this.destroyed = true
    })

    emit(event: string) {
      for (const handler of this.listeners.get(event) ?? []) {
        handler()
      }
    }

    constructor() {
      trayInstances.push(this)
    }
  }

  return {
    Menu: {
      buildFromTemplate: vi.fn((template: MenuItem[]) => ({ template }))
    },
    Tray: FakeTray,
    nativeImage: {
      createFromPath: vi.fn((path: string) => ({
        isEmpty: vi.fn(() => path === 'empty.png')
      }))
    },
    trayInstances
  }
})

vi.mock('electron', () => ({
  Menu: electronMock.Menu,
  Tray: electronMock.Tray,
  nativeImage: electronMock.nativeImage
}))

beforeEach(async () => {
  electronMock.trayInstances.length = 0
  vi.resetModules()
})

describe('tray', () => {
  it('creates a tray icon on first close and shows the window on click', async () => {
    const window = {
      focus: vi.fn(),
      hide: vi.fn(),
      isDestroyed: vi.fn(() => false),
      isFocused: vi.fn(() => false),
      isVisible: vi.fn(() => false),
      show: vi.fn()
    } as any

    const { ensureTray } = await import('./tray')
    ensureTray({ window, iconPath: 'valid.png', onQuit: vi.fn() })

    expect(electronMock.trayInstances).toHaveLength(1)
    const [tray] = electronMock.trayInstances
    expect(tray.tooltip).toBe('Hermes')
    expect(tray.menu?.map(item => item.label)).toEqual(['Show Hermes', undefined, 'Quit Hermes'])

    tray.emit('click')
    expect(window.show).toHaveBeenCalled()
    expect(window.focus).toHaveBeenCalled()
  })

  it('does not create a tray when the window is missing or destroyed', async () => {
    const onQuit = vi.fn()
    const { ensureTray } = await import('./tray')

    ensureTray({ window: null, iconPath: 'valid.png', onQuit })
    expect(electronMock.trayInstances).toHaveLength(0)

    const destroyedWindow = {
      focus: vi.fn(),
      hide: vi.fn(),
      isDestroyed: vi.fn(() => true),
      isFocused: vi.fn(() => false),
      isVisible: vi.fn(() => false),
      show: vi.fn()
    }

    ensureTray({ window: destroyedWindow as any, iconPath: 'valid.png', onQuit })
    expect(electronMock.trayInstances).toHaveLength(0)
  })

  it('does not create a tray when the icon path is missing or empty', async () => {
    const window = {
      focus: vi.fn(),
      hide: vi.fn(),
      isDestroyed: vi.fn(() => false),
      isFocused: vi.fn(() => false),
      isVisible: vi.fn(() => false),
      show: vi.fn()
    } as any

    const { ensureTray } = await import('./tray')

    ensureTray({ window, iconPath: undefined, onQuit: vi.fn() })
    ensureTray({ window, iconPath: 'empty.png', onQuit: vi.fn() })
    expect(electronMock.trayInstances).toHaveLength(0)
  })

  it('does not create a second tray if one already exists', async () => {
    const window = {
      focus: vi.fn(),
      hide: vi.fn(),
      isDestroyed: vi.fn(() => false),
      isFocused: vi.fn(() => false),
      isVisible: vi.fn(() => false),
      show: vi.fn()
    } as any

    const { ensureTray } = await import('./tray')
    ensureTray({ window, iconPath: 'valid.png', onQuit: vi.fn() })
    ensureTray({ window, iconPath: 'another.png', onQuit: vi.fn() })

    expect(electronMock.trayInstances).toHaveLength(1)
  })

  it('calls onQuit when the tray menu quit item is clicked', async () => {
    const window = {
      focus: vi.fn(),
      hide: vi.fn(),
      isDestroyed: vi.fn(() => false),
      isFocused: vi.fn(() => false),
      isVisible: vi.fn(() => false),
      show: vi.fn()
    } as any

    const onQuit = vi.fn()
    const { ensureTray } = await import('./tray')
    ensureTray({ window, iconPath: 'valid.png', onQuit })

    const quitItem = electronMock.trayInstances[0].menu?.find(item => item.label === 'Quit Hermes')
    quitItem?.click?.()

    expect(onQuit).toHaveBeenCalled()
  })

  it('destroys the tray icon during teardown', async () => {
    const window = {
      focus: vi.fn(),
      hide: vi.fn(),
      isDestroyed: vi.fn(() => false),
      isFocused: vi.fn(() => false),
      isVisible: vi.fn(() => false),
      show: vi.fn()
    } as any

    const { ensureTray, destroyTray } = await import('./tray')
    ensureTray({ window, iconPath: 'valid.png', onQuit: vi.fn() })

    const [tray] = electronMock.trayInstances
    destroyTray()

    expect(tray.destroy).toHaveBeenCalled()
    expect(tray.destroyed).toBe(true)
  })
})
