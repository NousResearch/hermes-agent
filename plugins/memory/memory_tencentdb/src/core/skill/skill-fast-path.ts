/**
 * skill-fast-path — name-substring quick-match channel.
 *
 * Design: docs/design/2026-06-30-skill-router.md §4.2
 *
 * 与主通道（BM25 / embedding / hybrid via ISkillStore.searchSkills）并行执行的独立快速通道。
 * 当 skill name 长度 ≥ NAME_MATCH_MIN_LENGTH（默认 4），且用户 query（小写）包含该
 * skill name（子串匹配）时，该 skill 被快速通道命中。
 *
 * 合并规则（在 handleListing 中实现）：
 * - 快速通道命中的 skill 放到最终结果首位
 * - 主通道结果去重后拼接在后面
 * - 不进 reranker，避免打乱已 rerank 的主通道顺序
 *
 * 性能：纯内存字符串操作，无 I/O，无模型推理；80k skill 遍历 <5ms。
 *
 * DECISION (2026-06-30): 暂不接入。评估结论：
 *   - fast-path 需要先拉全集（name + description），VDB 模式下 HTTP 拉取
 *     1000+ skill 需 80-150ms（同 region）甚至更多，远超纯内存匹配的 <5ms。
 *   - 主通道 BM25/hybrid 对 name 字段权重天然高（FTS5 name 列 position 0），
 *     用户提到名称时主通道大概率已 top-K 命中。
 *   - 收益有限（仅服务"用户显式提到 skill 名"的边缘 case）但成本为
 *     2~6 倍查询（search + 全量 list + 内存匹配 vs 单次 search）。
 *   保留文件待后续数据证明召回不足时再重新评估。
 */
import type { Skill } from "./types.js";

export const DEFAULT_NAME_MATCH_MIN_LENGTH = 4;

/**
 * 返回 `skills` 中 `name` 长度 ≥ `minLength` 且 query（小写）包含该 name（小写）的 skill。
 * 空 query 或全空白短路返回 `[]`。
 */
export function nameMatchFastPath(
  query: string,
  skills: Skill[],
  minLength: number = DEFAULT_NAME_MATCH_MIN_LENGTH,
): Skill[] {
  const q = (query ?? "").toLowerCase().trim();
  if (q === "") return [];
  return skills.filter(
    (s) => s.name.length >= minLength && q.includes(s.name.toLowerCase()),
  );
}
