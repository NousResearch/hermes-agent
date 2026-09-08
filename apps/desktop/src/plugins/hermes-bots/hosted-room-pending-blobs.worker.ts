// String encoding and Blob construction for large inputs must not run on the UI thread.
self.onmessage = (event: MessageEvent<string[] | string>) => {
  if (typeof event.data === 'string') {
    const pending = JSON.parse(event.data)
    self.postMessage(pending && { ...pending, ...(pending.attachments ? {
      attachments: pending.attachments.map((attachment: { data: string }) => {
        if (typeof attachment.data !== 'string') {throw new Error('Saved hosted attachment bytes are unavailable')}

        return { ...attachment, data: new Blob([attachment.data]) }
      })
    } : {}) })
  } else {
    self.postMessage(event.data.map(data => new Blob([data])))
  }
}
