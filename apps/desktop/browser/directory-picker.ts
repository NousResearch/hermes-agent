import { connection, requestJson } from './transport'

/** Browser file inputs address the phone/laptop; projects address the backend. */
export async function selectServerDirectory(profile = connection().profile || 'default'): Promise<string[]> {
  const { translateNow } = await import('../src/i18n/runtime')
  const { cwd } = await requestJson<{ cwd: string }>({ path: '/api/fs/default-cwd', profile })
  const dialog = document.createElement('dialog')
  dialog.className = 'w-full max-w-xl rounded-lg border bg-background p-4 text-foreground shadow-lg'
  const heading = document.createElement('h2')
  heading.textContent = translateNow('sidebar.projects.addFolderTitle')
  heading.className = 'mb-3 text-lg font-semibold'
  const form = document.createElement('form')
  const input = document.createElement('input')
  input.type = 'text'
  input.value = cwd
  input.setAttribute('aria-label', translateNow('sidebar.projects.foldersLabel'))
  input.className = 'mb-3 w-full rounded border bg-background p-2'
  const list = document.createElement('div')
  list.className = 'mb-3 max-h-72 overflow-auto'
  const error = document.createElement('p')
  error.setAttribute('role', 'alert')
  const buttons = document.createElement('div')
  buttons.className = 'flex justify-end gap-3'
  const cancel = document.createElement('button')
  cancel.type = 'button'
  cancel.textContent = translateNow('common.cancel')
  const choose = document.createElement('button')
  choose.type = 'submit'
  choose.textContent = translateNow('common.choose')
  choose.className = 'rounded border px-3 py-1'
  buttons.append(cancel, choose)
  form.append(input, list, error, buttons)
  dialog.append(heading, form)
  document.body.append(dialog)
  let generation = 0
  let selected: string | undefined

  const browse = async (path: string) => {
    const current = ++generation
    selected = undefined
    choose.disabled = true

    try {
      const result = await requestJson<{ entries: Array<{ name: string; path: string; isDirectory: boolean }>; error?: string }>({ path: `/api/fs/list?path=${encodeURIComponent(path)}`, profile })

      if (current !== generation || !dialog.isConnected) {return}

      if (result.error) {throw new Error(result.error)}
      input.value = path
      selected = path
      choose.disabled = false
      list.replaceChildren()
      error.textContent = ''
      const parent = path.replace(/[\\/]+$/, '').replace(/[\\/][^\\/]*$/, '') || '/'

      for (const entry of [{ name: '..', path: parent, isDirectory: true }, ...result.entries.filter(entry => entry.isDirectory)]) {
        const button = document.createElement('button')
        button.type = 'button'
        button.textContent = entry.name
        button.className = 'block w-full rounded px-2 py-1 text-left hover:bg-muted'

        button.onclick = () => { void browse(entry.path) }
        list.append(button)
      }
    } catch (failure) {
      if (current === generation) {error.textContent = failure instanceof Error ? failure.message : String(failure)}
    }
  }

  try {
    return await new Promise<string[]>(resolve => {
      cancel.onclick = () => dialog.close()
      dialog.addEventListener('close', () => resolve(dialog.returnValue === 'choose' && selected ? [selected] : []), { once: true })
      input.addEventListener('change', () => { void browse(input.value) })

      form.onsubmit = event => { event.preventDefault();

 if (selected && selected === input.value) {dialog.close('choose');} else {void browse(input.value)} }

      dialog.showModal()
      void browse(cwd)
    })
  } finally {
    generation += 1
    dialog.remove()
  }
}
