import '../styles.css'

import { createRoot } from 'react-dom/client'

import { MessageTextContent } from '@/components/assistant-ui/markdown-text'

const root = createRoot(document.getElementById('root')!)
let mount = 0

Object.assign(window, {
  renderImage(src: string) {
    root.render(
      <div data-slot="aui_thread-content" style={{ width: 'min(640px, 100%)' }}>
        <div data-slot="aui_assistant-message-content">
          <MessageTextContent
            key={++mount}
            text={`${src.includes('/tall.') ? `MEDIA:${src}` : `![Synthetic contact sheet](${src})`}\n\nFollowing text must remain fully readable below the image.`}
          />
        </div>
      </div>
    )
  },
  unmountImage() {
    root.render(null)
  }
})
