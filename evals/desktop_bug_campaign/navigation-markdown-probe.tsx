import '@/styles.css'
import { createRoot } from 'react-dom/client'
import type { ReactNode } from 'react'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from '@/lib/query-client'
import { I18nProvider } from '@/i18n'
import { ThemeProvider } from '@/themes/context'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import { assistantTextPart, renderMediaTags } from '@/lib/chat-messages/parts'
import { stripGeneratedImageEchoes } from '@/lib/generated-images'
import { extractEmbeddedImages } from '@/lib/embedded-images'
import { stripPreviewTargets } from '@/lib/preview-targets'
const samples = {
  hard: 'First line  \nSecond line',
  soft: 'Soft first\nSoft second',
  indented: '    value = 1\n\n',
  unfinished: '```python\nvalue = 1  ',
  code: '```python\nvalue = 1  \nvalue = 2 \n```',
  blanks: '```python\n\nvalue = 1  \n\n\n```'
}
const image = 'data:image/png;base64,' + 'A'.repeat(64)
const ingress = {
  assistant: (text: string) => { const part = assistantTextPart(text); return part.type === 'text' ? part.text : '' },
  media: (text: string) => renderMediaTags('MEDIA: /tmp/image.png\n\n' + text),
  preview: (text: string) => stripPreviewTargets('[Preview: x](#preview:test)\n\n' + text),
  embedded: (text: string) => extractEmbeddedImages(image + '\n\n' + text).cleanedText,
  generated: (text: string) => stripGeneratedImageEchoes('![result](/tmp/image.png)\n\n' + text, ['/tmp/image.png'])
}
const cases = Object.fromEntries(Object.entries(ingress).flatMap(([name, apply]) => Object.entries(samples).map(([kind, input]) => [`${name}-${kind}`, {kind, input, text: apply(input)}])))
Object.assign(window, {markdownCases: cases, preprocessMarkdown})
function Providers({children}: {children: ReactNode}) {
  return <QueryClientProvider client={queryClient}><I18nProvider><ThemeProvider><RootTooltipProvider>{children}</RootTooltipProvider></ThemeProvider></I18nProvider></QueryClientProvider>
}
createRoot(document.getElementById('root')!).render(<Providers><main style={{padding:40}}><h1>Production Markdown renderer probe</h1>{Object.entries(cases).map(([id, {text}]) => <section id={id} key={id}><h2>{id}</h2><MarkdownTextContent text={text} isRunning={false}/></section>)}</main></Providers>)
