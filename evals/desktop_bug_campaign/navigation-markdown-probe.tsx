import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from '@/lib/query-client'
import { I18nProvider } from '@/i18n'
import { ThemeProvider } from '@/themes/context'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
const hard = 'First line  \nSecond line'
const soft = 'Soft first\nSoft second'
const code = '```python\nvalue = 1  \nvalue = 2 \n```'
Object.assign(window, {probeInputs:{hard,soft,code}, probeOutputs:{hard:preprocessMarkdown(hard),soft:preprocessMarkdown(soft),code:preprocessMarkdown(code)}})
createRoot(document.getElementById('root')!).render(<QueryClientProvider client={queryClient}><I18nProvider><ThemeProvider><RootTooltipProvider><main style={{padding:40}}><h1>Production Markdown renderer probe</h1>{Object.entries({hard,soft,code}).map(([id,text])=><section id={id} key={id}><h2>{id}</h2><MarkdownTextContent text={text} isRunning={false}/></section>)}</main></RootTooltipProvider></ThemeProvider></I18nProvider></QueryClientProvider>)
