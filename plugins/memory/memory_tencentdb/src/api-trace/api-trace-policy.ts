/**
 * API trace 档位策略：由 metadata store backend 推断，无独立配置项。
 */
import type { MetadataBackend } from "../metadata/store/interface.js";

export type ApiTraceProfile = "full" | "lite";

export interface ApiTracePolicy {
  profile: ApiTraceProfile;
  module: string;
  maxFieldChars: number;
  maxJsonChars: number;
  maxSqlChars: number;
  /** 成功路径是否记录 HTTP body */
  httpBodyOnSuccess: boolean;
  /** 成功路径是否记录 service enter/exit */
  serviceLayerOnSuccess: boolean;
  /** 成功路径是否记录 store enter/exit */
  storeLayerOnSuccess: boolean;
  /** 是否调用 trace.report（OTel/Langfuse） */
  httpOtelReport: boolean;
}

export function resolveProfile(backend: MetadataBackend = "sqlite"): ApiTraceProfile {
  return backend === "mongodb" ? "full" : "lite";
}

export function resolvePolicy(backend: MetadataBackend = "sqlite"): ApiTracePolicy {
  const full = resolveProfile(backend) === "full";
  return {
    profile: full ? "full" : "lite",
    module: "meta",
    maxFieldChars: 1024,
    maxJsonChars: 8192,
    maxSqlChars: 2048,
    httpBodyOnSuccess: full,
    serviceLayerOnSuccess: full,
    storeLayerOnSuccess: full,
    httpOtelReport: full,
  };
}
