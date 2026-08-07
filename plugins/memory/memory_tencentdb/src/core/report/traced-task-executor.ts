/**
 * TracedTaskExecutor — 非侵入式 Trace 装饰器（门面层）
 *
 * 包装原始 TaskExecutor，为每个 L1/L2/L3 任务执行创建 OTel Span，
 * 并从 TaskPayload.data 中恢复跨异步边界的 Trace Context。
 *
 * 使用方式（在 server.ts 中）：
 *   const rawExecutor = this.buildTaskExecutor();
 *   const tracedExecutor = new TracedTaskExecutor(rawExecutor);
 *   this.pipelineWorker = new PipelineWorker(backend, tracedExecutor, ...);
 *
 * 不修改任何业务代码，纯可观测性组件。
 * 公开 API 签名保持不变，调用方无需修改。
 */

import type { TaskPayload } from "../state/types.js";
import type { TaskExecutor } from "../../services/pipeline-worker.js";
import { getObservabilityBackend } from "./factory.js";
import { obsLogger } from "./obs-logger.js";

/** 任务类型 → Span Name 映射 */
const TASK_SPAN_NAMES: Record<string, string> = {
  L1: "core.l1.extraction",
  L2: "core.l2.extraction",
  L3: "core.l3.generation",
  flush: "core.flush",
};

/**
 * TracedTaskExecutor — 装饰器模式包装 TaskExecutor。
 *
 * 对每个 executeL1/L2/L3 调用：
 * 1. 从 task.data 反序列化恢复上游 Trace Context
 * 2. 创建 CONSUMER 类型 Span（follow-from link）
 * 3. 在 Span context 中执行原始 executor
 * 4. 记录 instance_id、session_id、task_type 等业务属性
 * 5. 错误时设置 Span Error 状态
 */
export class TracedTaskExecutor implements TaskExecutor {
  private readonly inner: TaskExecutor;

  constructor(inner: TaskExecutor) {
    this.inner = inner;
  }

  async executeL1(task: TaskPayload): Promise<void> {
    return this.executeWithTrace("L1", task, () => this.inner.executeL1(task));
  }

  async executeL2(task: TaskPayload): Promise<void> {
    return this.executeWithTrace("L2", task, () => this.inner.executeL2(task));
  }

  async executeL3(task: TaskPayload): Promise<void> {
    return this.executeWithTrace("L3", task, () => this.inner.executeL3(task));
  }

  async executeFlush?(task: TaskPayload): Promise<void> {
    if (this.inner.executeFlush) {
      return this.executeWithTrace("flush", task, () => this.inner.executeFlush!(task));
    }
    return this.executeL1(task);
  }

  async executeOffloadL1?(task: TaskPayload, signal?: AbortSignal): Promise<void> {
    if (this.inner.executeOffloadL1) {
      return this.executeWithTrace("offload-l1", task, () => this.inner.executeOffloadL1!(task, signal));
    }
  }

  async executeOffloadL15?(task: TaskPayload, signal?: AbortSignal): Promise<void> {
    if (this.inner.executeOffloadL15) {
      return this.executeWithTrace("offload-l15", task, () => this.inner.executeOffloadL15!(task, signal));
    }
  }

  async executeOffloadL2?(task: TaskPayload, signal?: AbortSignal): Promise<void> {
    if (this.inner.executeOffloadL2) {
      return this.executeWithTrace("offload-l2", task, () => this.inner.executeOffloadL2!(task, signal));
    }
  }

  /**
   * 核心方法：在 Trace Context 中执行任务。
   *
   * 优先使用 traceMiddleware.withSpan() 让 fn 在 active span context 中执行，
   * 使得 fn 内部调用 metricProducer.send() 时能通过 serializeTraceContext()
   * 自动获取当前 span 的 traceId。
   *
   * 降级策略：若 withSpan 不可用，回退到 trace.start()/end() 模式。
   */
  private async executeWithTrace(
    taskType: string,
    task: TaskPayload,
    fn: () => Promise<void>,
  ): Promise<void> {
    const backend = getObservabilityBackend();
    const spanName = TASK_SPAN_NAMES[taskType] ?? `core.task.${taskType.toLowerCase()}`;

    // 提取业务属性
    const instanceId = task.instanceId
      ?? (typeof task.data?.instanceId === "string" ? task.data.instanceId : "unknown");
    const sessionId = task.sessionId ?? "unknown";

    const attrs: Record<string, string | number | boolean> = {
      "instance_id": instanceId,
      "session_id": sessionId,
      "task_type": taskType,
      "task_id": task.id,
      "event_name": spanName,
      "messaging.system": "redis_stream",
      "messaging.operation": "process",
    };

    // 优先路径：使用 withSpan 让 fn 在 active span context 中执行
    if (typeof backend.traceMiddleware?.withSpan === "function") {
      return backend.traceMiddleware.withSpan(spanName, attrs, async (span) => {
        try {
          await fn();
          span.setStatus({ code: 0 /* SpanStatusCode.OK */ });
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          span.setStatus({ code: 2 /* SpanStatusCode.ERROR */, message: errMsg });
          span.recordException(err instanceof Error ? err : new Error(errMsg));

          obsLogger.error(`core.${taskType.toLowerCase()}.failed`, {
            instance_id: instanceId,
            session_id: sessionId,
            task_type: taskType,
            task_id: task.id,
            error: errMsg,
          }, err instanceof Error ? err : undefined);

          throw err;
        }
      });
    }

    // 降级路径：withSpan 不可用时，回退到 start/end（不激活 context）
    const span = backend.trace.start(spanName, 4 /* SpanKind.CONSUMER */);
    span.setAttributes(attrs);

    try {
      await fn();
      span.setStatus({ code: 0 /* SpanStatusCode.UNSET → OK */ });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      span.setStatus({ code: 2 /* SpanStatusCode.ERROR */, message: errMsg });
      span.recordException(err instanceof Error ? err : new Error(errMsg));

      obsLogger.error(`core.${taskType.toLowerCase()}.failed`, {
        instance_id: instanceId,
        session_id: sessionId,
        task_type: taskType,
        task_id: task.id,
        error: errMsg,
      }, err instanceof Error ? err : undefined);

      throw err;
    } finally {
      span.end();
    }
  }
}
