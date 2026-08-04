/**
 * Scene navigation: generates a summary navigation section appended to persona.md.
 *
 * The navigation includes **absolute** file paths so the agent can directly
 * use read_file for on-demand scene loading (progressive disclosure).
 */

import path from "node:path";
import type { SceneIndexEntry } from "./scene-index.js";

const NAV_HEADER = "---\n## 🗺️ Scene Navigation (Scene Index)";

const NAV_FOOTER_LOCAL = `📌 使用说明：
- Path 是 scene block 的绝对路径，可直接使用 **read** 工具读取完整内容（参数: filePath）
- 热度：该场景被记忆命中的累计次数，越高越重要
- Summary：场景的核心要点摘要`;

const NAV_FOOTER_COS = `📌 使用说明：
- Path 是 scene block 的存储路径，请使用 **tdai_read_cos** 工具读取完整内容（参数: path）
- 热度：该场景被记忆命中的累计次数，越高越重要
- Summary：场景的核心要点摘要`;

/**
 * Build a fire-emoji string based on heat value (visual priority cue for the agent).
 */
function heatEmoji(heat: number): string {
  if (heat >= 1000) return " 🔥🔥🔥🔥🔥";
  if (heat >= 500) return " 🔥🔥🔥🔥";
  if (heat >= 200) return " 🔥🔥🔥";
  if (heat >= 100) return " 🔥🔥";
  if (heat >= 50) return " 🔥";
  return "";
}

/**
 * Generate the scene navigation Markdown section.
 *
 * @param entries - Scene index entries
 * @param dataDir - Absolute path to the plugin data directory; when provided
 *                  and useCos=false, paths are absolute for read_file.
 * @param useCos  - When true, paths use scenes/ prefix and footer says tdai_read_cos.
 */
export function generateSceneNavigation(entries: SceneIndexEntry[], dataDir?: string, useCos = false): string {
  if (entries.length === 0) return "";

  const sorted = [...entries].sort((a, b) => b.heat - a.heat);

  const blocks = sorted.map((e) => {
    let scenePath: string;
    if (useCos) {
      scenePath = `scenes/${e.filename}`;
    } else {
      scenePath = dataDir
        ? path.join(dataDir, "scene_blocks", e.filename)
        : `scene_blocks/${e.filename}`;
    }
    const pathLine = `### Path: ${scenePath}`;
    const heatLine = `**热度**: ${e.heat}${heatEmoji(e.heat)}${e.updated ? ` | **更新**: ${e.updated}` : ""}`;
    const summaryLine = `Summary: ${e.summary}`;
    return `${pathLine}\n${heatLine}\n${summaryLine}`;
  });

  const toolHint = useCos ? "tdai_read_cos" : "read";
  const footer = useCos ? NAV_FOOTER_COS : NAV_FOOTER_LOCAL;

  return `${NAV_HEADER}\n*以下是当前场景记忆的索引，可根据需要 ${toolHint} 读取详细内容。*\n\n${blocks.join("\n\n")}\n\n${footer}`;
}

/**
 * Strip the scene navigation section from persona content.
 */
export function stripSceneNavigation(personaContent: string): string {
  const idx = personaContent.indexOf(NAV_HEADER);
  if (idx === -1) return personaContent;
  return personaContent.slice(0, idx).trimEnd();
}
