/**
 * 外部资产 ID 生成（设计 §4.1.1 / API 参考 AssetEntity）。
 * 元数据模块不生成 asset_id；本工具供调用方与测试使用。
 */
import type { AssetType } from "../types.js";
import { generateRelationId } from "./id-generator.js";

/** `asset_type` → `asset_id` 前缀（含后续 `-` 前的部分）。 */
export const EXTERNAL_ASSET_ID_PREFIX: Record<AssetType, string> = {
  skill: "skl",
  llm_wiki: "wiki",
  code_graph: "cg",
  chat_memory: "mem",
};

/** 生成符合规范的外部资产 ID，如 `skl-a3b9c1f2`。 */
export function newExternalAssetId(assetType: AssetType): string {
  return `${EXTERNAL_ASSET_ID_PREFIX[assetType]}-${generateRelationId()}`;
}
