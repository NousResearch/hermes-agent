export interface FieldCopyTree {
  [key: string]: string | FieldCopyTree
}

function isFieldCopyTree(value: unknown): value is FieldCopyTree {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function defineFieldCopy(copy: FieldCopyTree): Record<string, string> {
  const result: Record<string, string> = {}

  const visit = (node: FieldCopyTree, prefix: string[] = []) => {
    for (const [key, value] of Object.entries(node)) {
      const parts = key.split('.')

      if (parts.some(part => part.length === 0)) {
        throw new Error(`Invalid field copy key: ${[...prefix, key].join('.')}`)
      }

      const path = [...prefix, ...parts]

      if (typeof value === 'string') {
        const flatKey = path.join('.')

        if (Object.prototype.hasOwnProperty.call(result, flatKey)) {
          throw new Error(`Duplicate field copy key: ${flatKey}`)
        }

        result[flatKey] = value
        continue
      }

      if (!isFieldCopyTree(value)) {
        throw new Error(`Invalid field copy value for key: ${path.join('.')}`)
      }

      visit(value, path)
    }
  }

  visit(copy)

  return result
}
