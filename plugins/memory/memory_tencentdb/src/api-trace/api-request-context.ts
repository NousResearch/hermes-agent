/**
 * HTTP API trace 请求上下文（AsyncLocalStorage）。
 */
import { AsyncLocalStorage } from "node:async_hooks";

export interface ApiRequestContext {
  requestId: string;
  route: string;
  module: string;
  instanceId?: string;
  userId?: string;
  internal?: boolean;
}

export const apiRequestStorage = new AsyncLocalStorage<ApiRequestContext>();

export function runWithApiRequestContext<T>(ctx: ApiRequestContext, fn: () => T): T {
  return apiRequestStorage.run(ctx, fn);
}

export function getApiRequestContext(): ApiRequestContext | undefined {
  return apiRequestStorage.getStore();
}
