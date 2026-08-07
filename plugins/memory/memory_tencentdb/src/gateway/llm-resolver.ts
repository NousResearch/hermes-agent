/**
 * LLM provider 解析器 —— 将 yaml 中的 llm 段和 memory systemUser 组合成
 * 内核 runner 直接可用的 {baseUrl, apiKey, model, ...} 配置。
 *
 * 两种模式：
 *   1. provider="openai"（默认）：原样返回 llm 配置，向后兼容
 *   2. provider="proxy"：把 baseUrl 拼成 `${baseUrl}/proxy/<instanceId>/v1`，
 *      apiKey 使用 metadata.systemUser.memory.userKey（memory 系统角色 sk-mem-xxx）。
 *      这样内核所有 LLM 调用都会带上 memory 身份，走 context_proxy 的统一鉴权、
 *      成本守卫、可观测链路。
 *
 * 单一职责：本文件只做"计算最终 baseUrl / apiKey"，不新建 runner；四个接入点
 * （tdai-core.ts:636/987/1037/1082、server.ts buildOffloadLlmClient）在构造
 * runner 前统一调用本函数。
 */

import type { StandaloneLLMConfig } from "../adapters/standalone/llm-runner.js";
import type { GatewayMetadataConfig } from "./config.js";
import {
  isValidMemorySystemUserKey,
  resolveMemorySystemUserConfig,
  type MemorySystemUserConfig,
} from "../metadata/system-user.js";

export class LlmResolveError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LlmResolveError";
  }
}

/**
 * 计算给定 instanceId 下内核 LLM 调用的最终 (baseUrl, apiKey, model, ...)。
 *
 * @param llm            gateway yaml 中的 llm 段（含 provider）
 * @param instanceId     当前处理的实例 id（用于 provider=proxy 时拼路径）
 * @param memorySystemUser 已解析的 memory 系统用户配置（可选，仅 provider=proxy 且
 *                         useMemorySystemUserKey=true 时使用）
 *
 * @throws {LlmResolveError} provider=proxy 但缺失必要配置时
 */
export function resolveEffectiveLlmConfig(
  llm: StandaloneLLMConfig,
  instanceId: string | undefined,
  memorySystemUser: MemorySystemUserConfig | undefined,
): StandaloneLLMConfig {
  const provider = llm.provider ?? "openai";
  if (provider !== "proxy") {
    // 默认路径：原样返回，行为完全等价改造前
    return llm;
  }

  // provider=proxy 的校验
  if (!llm.baseUrl) {
    throw new LlmResolveError(
      "llm.provider=proxy 需要 llm.baseUrl 指向 context_proxy 根 URL (如 http://127.0.0.1:8096)",
    );
  }
  if (!instanceId || !instanceId.trim()) {
    throw new LlmResolveError(
      "llm.provider=proxy 需要 instanceId，但 core 当前 instanceId 为空 —— " +
      "service 模式下确保请求带 x-tdai-service-id，standalone 模式下确保 yaml 有 instanceId",
    );
  }

  const useSystemUserKey = llm.proxy?.useMemorySystemUserKey ?? true;
  let effectiveApiKey = llm.apiKey;
  if (useSystemUserKey) {
    if (!memorySystemUser) {
      throw new LlmResolveError(
        "llm.provider=proxy 且 llm.proxy.useMemorySystemUserKey=true 需要 " +
        "metadata.systemUser.memory 完整配置（userId + userKey），当前缺失",
      );
    }
    if (!isValidMemorySystemUserKey(memorySystemUser.userKey)) {
      throw new LlmResolveError(
        "metadata.systemUser.memory.userKey 必须匹配 sk-mem-[A-Za-z0-9_-]{32}",
      );
    }
    effectiveApiKey = memorySystemUser.userKey;
  }

  if (!effectiveApiKey) {
    throw new LlmResolveError(
      "llm.provider=proxy 且 useMemorySystemUserKey=false 时必须显式配置 llm.apiKey",
    );
  }

  // baseUrl 拼接规则：去掉尾部斜杠，追加 /proxy/<iid>/v1
  const cleanBase = llm.baseUrl.replace(/\/+$/, "");
  const proxyBaseUrl = `${cleanBase}/proxy/${encodeURIComponent(instanceId)}/v1`;

  return {
    ...llm,
    baseUrl: proxyBaseUrl,
    apiKey: effectiveApiKey,
  };
}

/**
 * 只做校验、不返回配置 —— 用于启动期"fail-fast"检查。
 * standalone 模式下 instanceId 一般是 "default"，可以直接校验；
 * service 模式下 instanceId 每次请求才知道，启动期只校验 systemUser 是否合法。
 */
export function validateLlmProviderConfig(
  llm: StandaloneLLMConfig,
  metadata: GatewayMetadataConfig,
): void {
  if (llm.provider !== "proxy") return;

  if (!llm.baseUrl) {
    throw new LlmResolveError(
      "llm.provider=proxy 需要 llm.baseUrl 指向 context_proxy 根 URL",
    );
  }

  const useSystemUserKey = llm.proxy?.useMemorySystemUserKey ?? true;
  if (useSystemUserKey) {
    const memoryUser = resolveMemorySystemUserConfig(metadata);
    if (!memoryUser) {
      throw new LlmResolveError(
        "llm.provider=proxy 且 useMemorySystemUserKey=true 需要 " +
        "metadata.systemUser.memory 完整配置（userId + userKey）",
      );
    }
    if (!isValidMemorySystemUserKey(memoryUser.userKey)) {
      throw new LlmResolveError(
        "metadata.systemUser.memory.userKey 必须匹配 sk-mem-[A-Za-z0-9_-]{32}",
      );
    }
  } else if (!llm.apiKey) {
    throw new LlmResolveError(
      "llm.provider=proxy 且 useMemorySystemUserKey=false 时必须显式配置 llm.apiKey",
    );
  }
}
