// First step of `dev:electron`, `profile:main` and `profile:main:cpu`: fetch get-windows' win32
// binding into node_modules (see ensureDevGetWindowsBinding).
import { ensureDevGetWindowsBinding } from './stage-native-deps.mjs'

ensureDevGetWindowsBinding()
