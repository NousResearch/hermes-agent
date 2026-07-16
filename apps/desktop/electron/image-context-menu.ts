export function imageContextMenuItems(params, actions) {
  if (params.mediaType !== 'image') {
    return []
  }

  const items = []
  const srcURL = params.srcURL || ''

  if (srcURL) {
    items.push({
      label: 'Open Image',
      click: () => {
        if (!srcURL.startsWith('data:')) {
          actions.openImage(srcURL)
        }
      },
      enabled: !srcURL.startsWith('data:')
    })
  }

  items.push({
    label: 'Copy Image',
    click: () => actions.copyImageAt(params.x, params.y)
  })

  if (srcURL) {
    items.push(
      {
        label: 'Copy Image Address',
        click: () => actions.copyImageAddress(srcURL)
      },
      {
        label: 'Save Image As...',
        click: () => actions.saveImage(srcURL)
      }
    )
  }

  return items
}
