/**
 * Core 结构化日志门面 — 基于 ILogBackend 抽象
 *
 * 提供 info/warn/error 方法，自动关联当前 Trace Context，
 * 日志通过 ILogBackend 后端上报（内部环境：智研 + ClickHouse）。
 *
 * 使用方式：
 *   import { obsLogger } from "../core/report/obs-logger.js";
 *   obsLogger.error("core.llm.timeout", { instance_id: "xxx", session_id: "yyy" });
 *
 * 不修改任何业务代码，纯可观测性组件。
 * 所有异常不影响业务启动和服务流程。
 * 同时写入本地日志文件（通过 FileLogger）。
 *
 * 公开 API 签名保持不变，调用方无需修改。
 */

import { FileLogger } from "./file-logger.js";
import { getObservabilityBackend } from "./factory.js";

export type LogAttrs = Record<string, string | number | boolean>;

// 初始化文件写入器（降级策略：初始化失败不影响业务）
const obsFileLogger = new FileLogger({
  path: process.env.LOG_PATH || "/data/log/",
  filename: "observability.log",
  rotateSizeBytes: 100 * 1024 * 1024, // 100MB
  rotateBackupLimit: 10,
});

/**
 * 可观测性日志门面。
 * 所有方法都是安全的（不抛异常、不阻塞业务）。
 */
export const obsLogger = {
  /**
   * INFO 级别日志 — 用于记录正常流程的关键节点。
   */
  info(eventName: string, attrs: LogAttrs = {}): void {
    try {
      getObservabilityBackend().log.info(eventName, attrs);
      // 同时写入本地日志文件
      obsFileLogger.write("INFO", eventName, attrs as Record<string, unknown>);
    } catch {
      // 静默失败，不影响业务
    }
  },

  /**
   * WARN 级别日志 — 用于记录可恢复的异常（如重试、降级）。
   */
  warn(eventName: string, attrs: LogAttrs = {}): void {
    try {
      getObservabilityBackend().log.warn(eventName, attrs);
      // 同时写入本地日志文件
      obsFileLogger.write("WARN", eventName, attrs as Record<string, unknown>);
    } catch {
      // 静默失败，不影响业务
    }
  },

  /**
   * ERROR 级别日志 — 用于记录不可恢复的错误（如 LLM 超时、VDB 写入失败）。
   */
  error(eventName: string, attrs: LogAttrs = {}, error?: Error): void {
    try {
      if (error) {
        attrs = { ...attrs, "error.message": error.message, "error.type": error.name };
      }
      getObservabilityBackend().log.error(eventName, attrs, error);
      // 同时写入本地日志文件
      obsFileLogger.write("ERROR", eventName, attrs as Record<string, unknown>);
    } catch {
      // 静默失败，不影响业务
    }
  },
};
