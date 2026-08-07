/**
 * Log 结构化日志门面 — Debug/Info/Warn/Error
 *
 * 使用方式：
 *
 *   import { log } from "./core/report/log.js";
 *
 *   log.info("recall completed", { count: 10, latencyMs: 42 });
 *   log.error("embedding failed", { provider: "zhipu", error: err.message });
 *
 * 底层通过 ILogBackend 发送结构化日志（内部环境：OTel Logs API → 智研 + ClickHouse）。
 * 自动关联当前 Trace 上下文（如果在 Span 内打日志，Log 自动带 TraceID/SpanID）。
 * 同时写入本地日志文件（通过 FileLogger）。
 * 如果后端未初始化，fallback 到 文件 + console。
 *
 * 公开 API 签名保持不变，调用方无需修改。
 */

import { FileLogger } from "./file-logger.js";
import { getObservabilityBackend } from "./factory.js";

// 初始化文件写入器（降级策略：初始化失败不影响业务）
const fileLogger = new FileLogger({
  path: process.env.LOG_PATH || "/data/log/",
  filename: "core.log",
  rotateSizeBytes: 100 * 1024 * 1024, // 100MB
  rotateBackupLimit: 10,
});

/**
 * 发送一条结构化日志。
 * 通过 ILogBackend 上报，同时写入本地文件。
 */
function emit(level: "DEBUG" | "INFO" | "WARN" | "ERROR", message: string, data?: Record<string, unknown>): void {
  try {
    // 构建属性（只接受原始类型）
    const attrs: Record<string, string | number | boolean> = {};
    if (data) {
      for (const [key, value] of Object.entries(data)) {
        if (value === null || value === undefined) continue;
        if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
          attrs[key] = value;
        }
      }
    }

    // 通过 ILogBackend 上报
    const backend = getObservabilityBackend().log;
    switch (level) {
      case "DEBUG":
        backend.debug?.(message, attrs);
        break;
      case "INFO":
        backend.info(message, attrs);
        break;
      case "WARN":
        backend.warn(message, attrs);
        break;
      case "ERROR":
        backend.error(message, attrs);
        break;
    }

    // 同时写入本地日志文件（无论后端是否可用）
    fileLogger.write(level, message, data);
  } catch {
    // 不阻塞业务
  }
}

export const log = {
  debug(message: string, data?: Record<string, unknown>): void {
    emit("DEBUG", message, data);
  },

  info(message: string, data?: Record<string, unknown>): void {
    emit("INFO", message, data);
  },

  warn(message: string, data?: Record<string, unknown>): void {
    emit("WARN", message, data);
  },

  error(message: string, data?: Record<string, unknown>): void {
    emit("ERROR", message, data);
  },
};
