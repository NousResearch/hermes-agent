// 技能显示层汉化（只做展示，不动技能标识、正文与后端 description）。
//
// 对照表 src/i18n/skill-catalog.zh.json 是离线生成的静态文件：键是技能名
// （即 `/name`、mutation、queryKey 里用的原标识），值是中文显示名、中文说明、
// 生成时依据的英文原文以及分类。技能名本身永远不变，变了会打断斜杠补全与
// 后端调用。
import skillCatalog from '@/i18n/skill-catalog.zh.json'

export interface SkillCatalogEntry {
  category: string | null
  description: string
  displayName: string
  /** 译文的依据：上游 description 改了，就回退原文，避免显示对不上的旧译文。 */
  sourceDescription: string
}

const CATEGORIES = skillCatalog.categories as Record<string, string>
const SKILLS = skillCatalog.skills as Record<string, SkillCatalogEntry>

/** 桌面 locale 的值是 'zh'（见 i18n/languages.ts），不是 'zh-CN'。 */
const isChinese = (locale: string): boolean => locale === 'zh' || locale === 'zh-hant'

/** 列表/详情标题：中文名后面保留原标识，用户仍能认出 `/pdf` 这样的调用名。 */
export function skillDisplayName(name: string, locale: string): string {
  if (!isChinese(locale)) {
    return name
  }

  return SKILLS[name]?.displayName ?? name
}

/** 说明文字：仅当传入的原文与译文依据一致时才替换，否则原样返回。 */
export function skillDescription(name: string, original: string, locale: string): string {
  if (!isChinese(locale)) {
    return original
  }

  const entry = SKILLS[name]

  return entry && entry.sourceDescription === original ? entry.description : original
}

/** 斜杠补全行：入参是补全命令（形如 `/omh-browser`），键要还原成技能名再查表。 */
export function skillCommandDescription(command: string, original: string, locale: string): string {
  const name = command.replace(/^\//, '').trim().split(/\s+/)[0] ?? ''

  return name ? skillDescription(name, original, locale) : original
}

export function skillCategoryLabel(category: string, locale: string): string {
  if (!isChinese(locale)) {
    return category
  }

  return CATEGORIES[category] ?? category
}

/** 检索词：中英都进，中文说明能搜到，英文原名/英文说明的检索不受影响。 */
export function skillSearchText(
  skill: { category?: unknown; description?: unknown; name?: unknown },
  locale: string
): string {
  const name = typeof skill.name === 'string' ? skill.name : ''
  const description = typeof skill.description === 'string' ? skill.description : ''
  const category = typeof skill.category === 'string' ? skill.category : ''
  const parts = [name, description, category]

  if (isChinese(locale)) {
    const entry = SKILLS[name]
    parts.push(entry?.displayName ?? '', entry?.description ?? '', CATEGORIES[category] ?? '')
  }

  return parts.filter(Boolean).join('\n')
}
