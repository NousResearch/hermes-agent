// A checkout reset is the start of preparation, not its completion.
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

// The relaunched app consumes (unlinks) the result file as soon as it boots, so
// a file that was listed a moment ago can be gone by the time it is read.
function readIfPresent(file) {
  try { return readFileSync(file, 'utf8') } catch (error) {
    if (error?.code === 'ENOENT') return null
    throw error
  }
}

export function observeSourceUpdate({ home, resultPath, expectSha }) {
  const directory = join(home, 'logs/update_receipts')
  const files = () => [
    ...(resultPath && existsSync(resultPath) ? [resultPath] : []),
    ...(existsSync(directory) ? readdirSync(directory)
      .filter(name => name.startsWith('update_') && name.endsWith('.json'))
      .map(name => join(directory, name)) : []),
  ]
  const before = new Map(files().map(file => [file, readIfPresent(file)]))
  return head => {
    let complete = false
    for (const file of files()) {
      const text = readIfPresent(file)
      if (text === null || before.get(file) === text) continue
      let data
      try { data = JSON.parse(text.replace(/^\uFEFF/, '')) } catch { continue }
      if (file === resultPath) {
        if (data.ok !== true || (data.exit_code != null && data.exit_code !== 0)) {
          throw new Error(`update handoff failed: ${text}`)
        }
        complete = true
      } else if (data.finished_at) {
        if (data.outcome !== 'success') throw new Error(`source update failed: ${text}`)
        if (!expectSha || data.post_update?.sha === expectSha) complete = true
      }
    }
    return complete && (!expectSha || head === expectSha)
      && !existsSync(join(home, '.hermes-update-in-progress'))
  }
}