// Chinese labels for the skill CATEGORY tabs + group headers on the 技能
// (Skills) page. Same rationale — and same boundaries — as
// skill-descriptions-zh.ts: the upstream skill assets are OURS-to-display-only,
// so the SKILL.md folders and the runtime (`tools/skills_tool.py`) stay
// untouched. The label is swapped in at render time in app/skills/index.tsx.
//
// Keyed by the category string the runtime derives from a skill's PARENT FOLDER
// (`tools/skills_tool.py` → `_get_category_from_path` returns the first path
// segment, e.g. "productivity", "mlops"); skills with no category folder land in
// the default bucket "general".
//
// WHITELIST — never machine-translated. A category with no entry falls back to
// `prettyName()` (the English folder name, title-cased), so an unmapped or
// newly-added category still renders sensibly and we never invent a translation.
//
// BRAND / TERM keys keep their canonical form on purpose: Apple, GitHub, MLOps,
// DevOps stay Latin; 元宝 (Tencent Yuanbao) is itself a Chinese product name.
// Only descriptive categories are translated.
//
// A few keys are listed defensively so they render correctly regardless of how
// the runtime groups skills: the `mlops/*` sub-groups, plus `dogfood` / `yuanbao`
// (which currently sit in the "general" bucket). If the runtime never emits one,
// its entry is simply inert.
//
// MAINTENANCE: keep the two maps in sync. A key present in one script but absent
// in the other falls back to English for the missing locale (safe, but visibly
// out of place) — add BOTH the Simplified and Traditional entry together.

export const CATEGORY_LABELS_ZH: Readonly<Record<string, string>> = {
  // ── brand / term: canonical form kept on purpose ──
  apple: 'Apple',
  devops: 'DevOps',
  github: 'GitHub',
  mlops: 'MLOps',
  'mlops/evaluation': 'MLOps · 评估',
  'mlops/inference': 'MLOps · 推理',
  'mlops/models': 'MLOps · 模型',
  yuanbao: '元宝', // Tencent Yuanbao — a Chinese product name

  // ── descriptive categories ──
  'autonomous-ai-agents': '自主 Agent',
  creative: '创意',
  'data-science': '数据科学',
  dogfood: '内部试用',
  email: '邮件',
  general: '通用',
  media: '媒体',
  'note-taking': '笔记',
  productivity: '效率',
  research: '研究',
  'smart-home': '智能家居',
  'social-media': '社交媒体',
  'software-development': '软件开发'
}

export const CATEGORY_LABELS_ZH_HANT: Readonly<Record<string, string>> = {
  // ── brand / term: canonical form kept on purpose ──
  apple: 'Apple',
  devops: 'DevOps',
  github: 'GitHub',
  mlops: 'MLOps',
  'mlops/evaluation': 'MLOps · 評估',
  'mlops/inference': 'MLOps · 推理',
  'mlops/models': 'MLOps · 模型',
  yuanbao: '元寶', // Tencent Yuanbao — a Chinese product name (Traditional)

  // ── descriptive categories ──
  'autonomous-ai-agents': '自主 Agent',
  creative: '創意',
  'data-science': '資料科學',
  dogfood: '內部試用',
  email: '郵件',
  general: '通用',
  media: '媒體',
  'note-taking': '筆記',
  productivity: '效率',
  research: '研究',
  'smart-home': '智慧家庭',
  'social-media': '社群媒體',
  'software-development': '軟體開發'
}

/**
 * Simplified-Chinese label for a skill category, keyed by the runtime category
 * string. Falls back to the provided label (prettyName's English) when the
 * category isn't in our whitelist — so unmapped / new categories still render.
 */
export function zhCategoryLabel(key: string, fallback: string): string {
  return CATEGORY_LABELS_ZH[key] ?? fallback
}

/**
 * Traditional-Chinese label for a skill category. Same whitelist contract as
 * zhCategoryLabel; falls straight back to the English label (never to the
 * Simplified map) so a Traditional locale never leaks Simplified glyphs.
 */
export function zhHantCategoryLabel(key: string, fallback: string): string {
  return CATEGORY_LABELS_ZH_HANT[key] ?? fallback
}
