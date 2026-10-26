import type { OpenDialogOptions } from 'electron'

type DialogProperty = NonNullable<OpenDialogOptions['properties']>[number]

// `hermes:selectPaths` → `dialog.showOpenDialog` properties. Directory pickers
// carry `createDirectory` so macOS shows New Folder: a new project needs a
// folder, and the picker is where the user expects to make one.
export function selectPathsDialogProperties(options: { directories?: boolean; multiple?: boolean } = {}) {
  const properties: DialogProperty[] = options.directories ? ['openDirectory', 'createDirectory'] : ['openFile']

  if (options.multiple !== false) {
    properties.push('multiSelections')
  }

  return properties
}
