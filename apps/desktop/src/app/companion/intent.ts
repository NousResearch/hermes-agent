export type CompanionIntent =
  | { kind: 'chat'; target: null }
  | { kind: 'find-file'; target: string }
  | { kind: 'open-app'; target: string }
  | { kind: 'open-folder'; target: string }

const FOLDER_TARGETS = new Set([
  'desktop',
  'documents',
  'downloads',
  'home',
  'music',
  'pictures',
  'videos',
  '桌面',
  '文档',
  '下载',
  '主页',
  '主目录',
  '音乐',
  '图片',
  '照片',
  '视频'
])

const OPEN_PATTERNS = [/^打开(.+)$/u, /^启动(.+)$/u]
const FIND_PATTERNS = [/^帮我找(.+)$/u, /^查找(.+)$/u, /^搜索(.+)$/u]

function extractTarget(input: string, patterns: RegExp[]): string | null {
  const trimmed = input.trim()

  for (const pattern of patterns) {
    const match = trimmed.match(pattern)

    if (match?.[1]) {
      return match[1].trim()
    }
  }

  return null
}

export function classifyCompanionIntent(input: string): CompanionIntent {
  const openTarget = extractTarget(input, OPEN_PATTERNS)

  if (openTarget) {
    if (FOLDER_TARGETS.has(openTarget.toLowerCase()) || FOLDER_TARGETS.has(openTarget)) {
      return { kind: 'open-folder', target: openTarget }
    }

    return { kind: 'open-app', target: openTarget }
  }

  const findTarget = extractTarget(input, FIND_PATTERNS)

  if (findTarget) {
    return { kind: 'find-file', target: findTarget }
  }

  return { kind: 'chat', target: null }
}
