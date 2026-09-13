import { installBrowserDesktopBridge } from './browser-desktop-bridge'

if (typeof window !== 'undefined') {
  installBrowserDesktopBridge()
}
