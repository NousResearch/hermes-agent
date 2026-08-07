/**
 * MetadataService / IMetadataStore 的 Proxy trace 包装。
 */
import type { MetadataService } from "../metadata/service/metadata-service.js";
import type { IMetadataStore } from "../metadata/store/interface.js";
import { getApiRequestContext } from "./api-request-context.js";
import { getApiTraceConfig, isApiTraceActive } from "./api-log-config.js";
import { logApiTrace } from "./api-trace-logger.js";
import { sanitizeApiPayload } from "./api-sanitize.js";

const SERVICE_SKIP = new Set(["constructor", "scopedInstanceId"]);
const STORE_SKIP = new Set(["constructor", "init", "close"]);

function summarizeValue(value: unknown, maxFieldChars: number): string {
  if (value === null || value === undefined) return String(value);
  if (typeof value === "string") return value.length > 64 ? `${value.slice(0, 64)}…` : value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    const json = JSON.stringify(sanitizeApiPayload(value, maxFieldChars));
    return json.length > 256 ? `${json.slice(0, 256)}…` : json;
  } catch {
    return "[object]";
  }
}

function summarizeArgs(args: unknown[], maxFieldChars: number): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  if (args.length > 0 && typeof args[0] === "object" && args[0] !== null && !Array.isArray(args[0])) {
    for (const [k, v] of Object.entries(args[0] as Record<string, unknown>).slice(0, 12)) {
      if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
        out[k] = typeof v === "string" ? summarizeValue(v, maxFieldChars) : v;
      }
    }
  } else if (args.length > 0) {
    out.arg0 = summarizeValue(args[0], maxFieldChars);
  }
  return out;
}

function resultHint(value: unknown): string | undefined {
  if (value === null || value === undefined) return String(value);
  if (typeof value === "object" && !Array.isArray(value)) {
    const o = value as Record<string, unknown>;
    for (const key of ["user_id", "team_id", "agent_id", "task_id", "asset_id", "key_id"]) {
      if (typeof o[key] === "string") return `${key}=${o[key]}`;
    }
  }
  if (Array.isArray(value)) return `array_len=${value.length}`;
  return undefined;
}

function wrapAsyncMethod(
  sourceFile: string,
  layer: "service" | "store",
  op: string,
  fn: (...args: unknown[]) => unknown,
): (...args: unknown[]) => unknown {
  return (...args: unknown[]) => {
    if (!isApiTraceActive() || !getApiRequestContext()) {
      return fn(...args);
    }

    const policy = getApiTraceConfig().policy;
    const layerOnSuccess =
      layer === "service" ? policy.serviceLayerOnSuccess : policy.storeLayerOnSuccess;
    const maxField = policy.maxFieldChars;
    const eventPrefix = layer === "service" ? "api.service" : "api.store";

    if (layerOnSuccess) {
      logApiTrace("info", `${eventPrefix}.enter`, {
        source_file: sourceFile,
        source_op: op,
        ...summarizeArgs(args, maxField),
      });
    }

    const started = Date.now();
    const finish = (result: unknown) => {
      if (!layerOnSuccess) return result;
      const hint = resultHint(result);
      const attrs: Record<string, string | number | boolean> = {
        source_file: sourceFile,
        source_op: op,
        duration_ms: Date.now() - started,
        success: true,
      };
      if (hint) attrs.result_hint = hint;
      logApiTrace("info", `${eventPrefix}.exit`, attrs);
      return result;
    };

    const onError = (err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      logApiTrace(
        "error",
        `${eventPrefix}.error`,
        {
          source_file: sourceFile,
          source_op: op,
          duration_ms: Date.now() - started,
          success: false,
          error_message: message,
          ...summarizeArgs(args, maxField),
        },
        { err: err instanceof Error ? err : undefined },
      );
      throw err;
    };

    try {
      const result = fn(...args);
      if (result instanceof Promise) {
        return result.then(finish).catch(onError);
      }
      return finish(result);
    } catch (err) {
      onError(err);
    }
  };
}

export function wrapApiServiceForTrace(svc: MetadataService): MetadataService {
  if (!isApiTraceActive()) return svc;

  return new Proxy(svc, {
    get(target, prop, receiver) {
      if (prop === "rawStore") {
        return Reflect.get(target, prop, receiver) as IMetadataStore;
      }
      const val = Reflect.get(target, prop, receiver);
      if (typeof val !== "function" || SERVICE_SKIP.has(String(prop))) return val;
      const op = String(prop);
      return wrapAsyncMethod("metadata-service.ts", "service", op, val.bind(target));
    },
  });
}

export function wrapApiStoreForTrace(
  store: IMetadataStore,
  sourceFile = "sqlite-adapter.ts",
): IMetadataStore {
  if (!isApiTraceActive()) return store;

  return new Proxy(store, {
    get(target, prop, receiver) {
      const val = Reflect.get(target, prop, receiver);
      if (typeof val !== "function" || STORE_SKIP.has(String(prop))) return val;
      const op = String(prop);
      return wrapAsyncMethod(sourceFile, "store", op, val.bind(target));
    },
  });
}
