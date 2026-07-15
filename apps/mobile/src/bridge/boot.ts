/**
 * boot.ts — side-effect module that installs the window.hermesDesktop bridge.
 *
 * Imported as the VERY FIRST import in main.tsx. ES module imports are hoisted
 * and run depth-first in source order, so importing this before any vendored
 * desktop module guarantees window.hermesDesktop exists before that vendored
 * code evaluates (some of it reads the bridge at module top-level).
 */
import { installBridge } from './install-bridge'

installBridge()
