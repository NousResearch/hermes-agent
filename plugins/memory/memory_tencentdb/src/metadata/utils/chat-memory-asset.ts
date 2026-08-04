/**
 * chat_memory 资产 ID 生成规则。
 *
 * 约定：每个 (team, agent) 组合对应**一个**稳定的 chat_memory 资产，asset_id
 * 由 team_id 与 agent_id 拼出：
 *
 *     chat_memory-{team_id}-{agent_id}
 *
 * 这种确定性 ID 让写入路径自动幂等 —— 同一 (team, agent) 无论请求多少次，
 * 计算出的 asset_id 相同，store 层的主键约束会自然拦截重复插入。
 *
 * 该 ID 不需要反解：调用方需要 team/agent 时是从上下文获取，不是从 ID 里
 * 拆出来。所以 team_id / agent_id 内部即使含有 `-` 也不影响正确性。
 */

/** 稳定生成一个 (team, agent) 对应的 chat_memory 资产 ID。 */
export function buildChatMemoryAssetId(teamId: string, agentId: string): string {
  return `chat_memory-${teamId}-${agentId}`;
}
