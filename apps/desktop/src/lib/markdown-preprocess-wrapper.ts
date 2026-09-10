import { tailBoundedRemend } from '@assistant-ui/react-streamdown'

import { preprocessMarkdown } from './markdown-preprocess'
import {
  createIncrementalMarkdownPreprocessor,
  type IncrementalMarkdownPreprocessor
} from './markdown-preprocess-cache'

type MarkdownTransform = (text: string) => string

export function createIncrementalPreprocessWithTailRepair(
  fullPreprocess: MarkdownTransform = preprocessMarkdown,
  repair: MarkdownTransform = tailBoundedRemend
): IncrementalMarkdownPreprocessor {
  const preprocessIncrementally = createIncrementalMarkdownPreprocessor(fullPreprocess)

  const preprocess: IncrementalMarkdownPreprocessor = (text: string): string => {
    try {
      return repair(preprocessIncrementally(text))
    } catch {
      return text
    }
  }

  preprocess.clear = preprocessIncrementally.clear

  return preprocess
}
