// Runtime bring-up/teardown: pen's transport socket + device manager,
// loaded lazily from the user's installed Pen.app on first canvas open.

import { findPenInstallation, PEN_SOCKET_APP_NAME, requirePenModule } from '../pen-host'

import { createLibraryDocument, penAutosaveTimers } from './documents'
import { documents, events, log, type PenRuntime, runtime, setPenRuntime } from './state'

// ---------------------------------------------------------------------------
// Runtime bring-up (lazy — first canvas open)
// ---------------------------------------------------------------------------

export function ensureRuntime(): PenRuntime | null {
  if (runtime) {
    return runtime
  }

  const install = findPenInstallation()

  if (!install) {
    return null
  }

  try {
    const shared = requirePenModule(install, '@ha/shared/dist/cjs/index.js')
    const ipcLib = requirePenModule(install, '@ha/ipc/dist/index.cjs')
    const mcpLib = requirePenModule(install, '@ha/mcp/dist/cjs/index.js')

    const penLogger = {
      debug: log.debug,
      info: log.debug,
      warn: log.warn,
      error: log.error
    }

    // The transport socket IS hermes's pen identity: pencil-hermes.sock.
    // Pen's MCP server binary (spawned by the agent bridge with
    // `--app hermes`) and `pen interactive -a hermes` both dial it.
    const transportServer = new ipcLib.TransportServerManager(penLogger, PEN_SOCKET_APP_NAME)

    const deviceManager = new ipcLib.IPCDeviceManager(
      transportServer,
      penLogger,
      install.unpackedPath,
      PEN_SOCKET_APP_NAME
    )

    transportServer.start()
    deviceManager.proxyMcpToolCallRequests()

    deviceManager.on?.('open-temporary-document', async (name: string) => {
      try {
        const info = await createLibraryDocument(name)
        events.emit('open-document', info)
      } catch (error) {
        log.warn('open-temporary-document failed', error)
      }
    })

    setPenRuntime({ install, shared, ipcLib, mcpLib, transportServer, deviceManager })
    log.info(`pen host up (Pen.app ${install.version || '?'}, socket app "${PEN_SOCKET_APP_NAME}")`)
  } catch (error) {
    log.warn('pen host bring-up failed — canvas unavailable', error)
    setPenRuntime(null)
  }

  return runtime
}

export function shutdownPenHost(): void {
  if (!runtime) {
    return
  }

  // Flush before teardown. This ran `documents.clear()` on quit with no save,
  // which is how unsaved canvas work was lost on restart — pen only writes on
  // an explicit ⌘S, so anything since the last one lived in memory and died
  // here. Synchronous so it completes inside the quit handler; a canvas is a
  // small JSON document, not a big write.
  for (const timer of penAutosaveTimers.values()) {
    clearTimeout(timer)
  }

  penAutosaveTimers.clear()

  for (const doc of documents.values()) {
    try {
      if (doc.device.getIsDirty()) {
        void doc.device.saveResource({ userAction: false })
      }
    } catch (error) {
      log.warn('save-on-quit failed', error)
    }
  }

  try {
    runtime.transportServer.stop()
  } catch {
    // socket already gone
  }

  documents.clear()
  setPenRuntime(null)
}
