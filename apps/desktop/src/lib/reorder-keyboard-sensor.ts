import { KeyboardSensor } from '@dnd-kit/core'

// A sortable row also owns menus and portaled dialogs. Their React key events
// still bubble through the row: only the focused activator owns reorder keys.
// Keep the stock sensor's key codes, movement, cancellation and focus restore.
export class ReorderKeyboardSensor extends KeyboardSensor {
  static activators = KeyboardSensor.activators.map(activator => ({
    ...activator,
    handler: (...args: Parameters<typeof activator.handler>) =>
      args[0].target === args[0].currentTarget && activator.handler(...args)
  }))
}
