import '../../styles.css'

import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { MarkdownPreview, SourceView } from '@/app/chat/right-rail/preview-file'

const text = Array.from({ length: 1600 }, (_, n) => `source line ${n + 1}`).join('\n')

function PreviewFocusFixture() {
  const [mode, setMode] = useState('source')
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    const update = () => setRevision(n => n + 1)
    window.addEventListener('preview-test-reload', update)

    return () => window.removeEventListener('preview-test-reload', update)
  }, [])

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', height: '100vh' }}>
      <div>
        <p data-selectable-text="true">Chat transcript outside the preview</p>
        <textarea aria-label="Composer" defaultValue="chat draft" />
        <button onClick={() => setMode('source')}>Show source</button>
        <button onClick={() => setMode('markdown')}>Show Markdown</button>
      </div>
      <div className="h-full min-h-0 overflow-auto">
        {mode === 'source' ? (
          <SourceView filePath="/workspace/file.txt" language="text" text={`${text}\nrevision ${revision}`} />
        ) : (
          <MarkdownPreview text={`# Markdown file\n\nRendered file body\n\nRevision ${revision}`} />
        )}
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<PreviewFocusFixture />)
