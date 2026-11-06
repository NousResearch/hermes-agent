import { createRoot } from 'react-dom/client'

import { MessageTextContent } from '@/components/assistant-ui/markdown-text'
import '../styles.css'

const root = createRoot(document.getElementById('root')!)
let mount = 0

Object.assign(window, {
  renderImage(src: string) {
    root.render(
      <div data-slot="aui_assistant-message-content" style={{ width: 'min(640px, 100%)' }}>
        <MessageTextContent
          key={++mount}
          text={`![Synthetic contact sheet](${src})\n\nFollowing text must remain fully readable below the image.`}
        />
      </div>
    )
  },
  unmountImage() {
    root.render(null)
  }
})
