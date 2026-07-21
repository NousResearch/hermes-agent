import { type BundledLanguage, getSingletonHighlighter } from 'shiki'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'

export interface ShikiWorkerRequest {
  code: string
  id: number
  language: string
}

export interface ShikiWorkerToken {
  content: string
  htmlStyle?: Record<string, string>
}

export interface ShikiWorkerResponse {
  error?: string
  id: number
  tokens?: ShikiWorkerToken[][]
}

const engine = createJavaScriptRegexEngine({ forgiving: true })

self.onmessage = async (event: MessageEvent<ShikiWorkerRequest>) => {
  const { code, id, language } = event.data

  try {
    const lang = (language || 'text') as BundledLanguage

    const highlighterOptions = {
      engine,
      langs: [lang],
      themes: ['github-dark-dimmed', 'github-light-default']
    }

    const highlighter = await getSingletonHighlighter(highlighterOptions)

    const result = highlighter.codeToTokens(code, {
      defaultColor: 'light-dark()',
      lang,
      themes: { dark: 'github-dark-dimmed', light: 'github-light-default' }
    })

    const tokens = result.tokens.map(line =>
      line.map(token => ({
        content: token.content,
        ...(token.htmlStyle ? { htmlStyle: token.htmlStyle as Record<string, string> } : {})
      }))
    )

    self.postMessage({ id, tokens } satisfies ShikiWorkerResponse)
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) } satisfies ShikiWorkerResponse)
  }
}
