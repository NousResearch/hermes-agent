/** Lock-owned recovery writes, adapted from kokhlo's #131067. */
export function createLaunchMarkerWriter() {
  const queued: Array<() => void> = []

  return {
    queue(write: () => void): void {
      queued.push(write)
    },
    flush(isPrimaryInstance: boolean): number {
      const writes = queued.splice(0)

      if (!isPrimaryInstance) {
        return 0
      }

      let written = 0

      for (const write of writes) {
        try {
          write()
          written++
        } catch (error) {
          console.warn('[hermes] could not persist a recovery marker:', error)
        }
      }

      return written
    }
  }
}
