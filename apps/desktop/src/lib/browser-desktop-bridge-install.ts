import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { installBrowserViewport } from './browser-viewport'

if (typeof window !== 'undefined' && installBrowserDesktopBridge()) {
  installBrowserViewport()
}
