import type { OpenDialogOptions } from 'electron'

type DialogProperty = NonNullable<OpenDialogOptions['properties']>[number]

// `hermes:selectPaths` → `dialog.showOpenDialog` properties. Directory pickers
// carry `createDirectory` so macOS shows New Folder: a new project needs a
// folder, and the picker is where the user expects to make one.
export function selectPathsDialogProperties(
  options: { directories?: boolean; multiple?: boolean } = {},
  platform = process.platform
) {
  const properties: DialogProperty[] = options.directories ? ['openDirectory'] : ['openFile']

  // Electron supports this option only on macOS. Passing it to the Windows
  // native folder dialog makes its navigation controls inaccessible.
  if (options.directories && platform === 'darwin') {
    properties.push('createDirectory')
  }

  if (options.multiple !== false) {
    properties.push('multiSelections')
  }

  return properties
}
