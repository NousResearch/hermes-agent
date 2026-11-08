/** Only a complete, explicit command can begin conversation preservation. */
export function closeoutCommand(text: string): { intent: 'archive' | 'new'; title?: string } | null {
  const command = text.trim()

  if (/^(?:\/?archive|archive this|archive the chat)$/i.test(command)) {
    return { intent: 'archive' }
  }

  const fresh = /^\/new(?:\s+([\s\S]+))?$/i.exec(command)

  return fresh ? { intent: 'new', ...(fresh[1]?.trim() ? { title: fresh[1].trim() } : {}) } : null
}
