/**
 * L1.5 Task Judgment Prompt — determines task lifecycle.
 */
import type { MmdMeta } from "../types.js";

export const L15_SYSTEM_PROMPT = `你是一个面向 AI 编码助手的"任务生命周期门神"。
你的职责是交叉分析提供的三个输入源，精准研判任务状态，并输出纯 JSON 对象。

【输入数据利用指南（必须遵循的思考链路）】
1. 第一步 - 剖析 recentMessages（识别意图）：根据当前和历史对话，提取用户最新回复的核心诉求。判断是"继续排查"、"宣布完工（如：跑通了）"、"单轮闲聊问答"还是"开启全新需求"。
2. 第二步 - 对齐 currentMmd（评估当前基线）：将用户的最新意图与 currentMmd 的完整 Mermaid 内容进行比对——关注 taskGoal、各节点的 status（done/doing/todo）以及 summary。如果诉求完全超出了当前图表的范畴或目标已实现（所有节点 done 且无后续），则 taskCompleted 为 true。若仍在解决图表中的子问题（包括 doing 节点或修 bug），则为 false。(如果没有currentMmd，就只根据当前对话和历史对话来判断是否继续任务)
3. 第三步 - 检索 availableMmds（判断是否延续）：如果判定要开启新任务（isLongTask=true 且 taskCompleted=true/当前无任务），必须扫描 availableMmds 的 taskGoal 和时间信息。若新诉求与列表中某个旧任务高度重合（如回到昨天没做完的模块），则是延续（isContinuation=true）。

【严格 JSON 输出格式】
务必输出合法的纯 JSON 对象，格式如下：
{
  "taskCompleted": boolean, // 当前任务是否已结束（如果 currentMmd 为 none，这里必须填 true）
  "isLongTask": boolean,    // 最新诉求是否是需要多步操作的复杂工程（普通技术问答、闲聊填 false）
  "isContinuation": boolean, // 是否在延续 availableMmds 中的历史任务
  "continuationMmdFile": "string|null", // 若延续旧任务，精确填入 availableMmds 中的文件名（不含路径前缀），否则为 null
  "newTaskLabel": "string|null" // 若是全新长任务，生成简短标签（≤30字符，kebab-case，如 "refactor-api"），否则为 null
}

只输出纯 JSON 对象，绝不允许包含解释文字。`;

export interface L15CurrentMmd {
  filename: string;
  content: string;
}

/**
 * Build the L1.5 user prompt for task judgment.
 */
export function buildL15UserPrompt(
  recentMessages: string,
  currentMmd: L15CurrentMmd | null,
  metas: MmdMeta[],
): string {
  const parts: string[] = [];

  parts.push("## 1. 最近的对话上下文 (Recent messages):");
  parts.push(recentMessages);
  parts.push("\n## 2. 当前挂载的任务图 (Active Mermaid — 完整内容):");

  if (currentMmd && currentMmd.filename) {
    parts.push(`**File:** ${currentMmd.filename}`);
    parts.push(`\n\`\`\`mermaid\n${currentMmd.content}\n\`\`\``);
  } else {
    parts.push("(none - 当前处于闲置状态，无活跃任务)");
  }

  parts.push("\n## 3. 历史可用的任务图 (Available Mermaid task files):");

  if (metas.length === 0) {
    parts.push("(none - 暂无历史长任务)");
  } else {
    for (const m of metas) {
      const total = m.doneCount + m.doingCount + m.todoCount;
      parts.push(`- **${m.filename}**`);
      parts.push(`  taskGoal: ${m.taskGoal}`);
      parts.push(
        `  progress: ${m.doneCount}/${total} done, ${m.doingCount} doing, ${m.todoCount} todo`,
      );
      if (m.updatedTime) {
        parts.push(`  lastUpdated: ${m.updatedTime}`);
      }
      if (m.nodeSummaries && m.nodeSummaries.length > 0) {
        parts.push("  recentNodes:");
        for (const n of m.nodeSummaries) {
          parts.push(`    - [${n.nodeId}] (${n.status}) ${n.summary}`);
        }
      }
      parts.push("");
    }
  }

  parts.push(
    "请严格根据系统指令的【三步思考链路】进行研判，并输出合法的 JSON 对象。",
  );
  return parts.join("\n");
}
