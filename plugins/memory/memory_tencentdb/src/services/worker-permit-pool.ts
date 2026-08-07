/**
 * WorkerPermitPool — memory PipelineWorker 并发信号量。
 *
 * 历史：曾经跨模块（memory PipelineWorker + skill 老 V2 worker）共用；
 * 2026-07-17 skill 改造后 skill 侧走 agent 级 extract-lock 做并发上限,
 * 不再依赖信号量。本 pool 目前只有 memory pipeline 一个 consumer,
 * 保留是因为 memory 侧 concurrency > 1 时仍需要它做队列。
 *
 * 语义：
 *   - capacity 是硬上限
 *   - acquire 满时排队；release 依 FIFO 唤醒
 *   - release 多于 acquire 抛错（帮助定位漏配对的 bug）
 *   - destroy 唤醒所有 waiting 使其 reject（用于优雅停机）
 */

const TAG = "[worker-permit-pool]";

export class WorkerPermitPool {
  readonly capacity: number;
  private _inFlight = 0;
  private readonly waiters: Array<{
    resolve: () => void;
    reject: (err: Error) => void;
  }> = [];
  private destroyed = false;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`${TAG} capacity must be a positive integer, got: ${capacity}`);
    }
    this.capacity = capacity;
  }

  /**
   * 获取一个许可。如果 in-flight 已达容量则挂起，直到有 release() 唤醒。
   * destroy 后 acquire 立即 reject。
   */
  acquire(): Promise<void> {
    if (this.destroyed) {
      return Promise.reject(new Error(`${TAG} pool destroyed`));
    }
    if (this._inFlight < this.capacity) {
      this._inFlight++;
      return Promise.resolve();
    }
    return new Promise<void>((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }

  /**
   * 归还一个许可。若有 waiter 则出队并 resolve（in-flight 计数不变）；
   * 否则 in-flight 递减。
   *
   * release 多于 acquire 视为编程错误，抛出以帮助定位泄漏点。
   */
  release(): void {
    if (this._inFlight <= 0 && this.waiters.length === 0) {
      throw new Error(`${TAG} release() called with no in-flight permit (unbalanced acquire/release)`);
    }
    const next = this.waiters.shift();
    if (next) {
      // 把 in-flight 顺延给 waiter，计数不变
      next.resolve();
    } else {
      this._inFlight--;
    }
  }

  /** 当前正在被持有的许可数。 */
  inFlight(): number {
    return this._inFlight;
  }

  /** 当前可立即拿到的许可数。 */
  available(): number {
    return Math.max(0, this.capacity - this._inFlight);
  }

  /** 当前排队等待的 acquire 数。 */
  waiting(): number {
    return this.waiters.length;
  }

  /**
   * 摧毁 pool：拒绝所有 waiter，之后 acquire 立即 reject。
   * 幂等。
   */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const err = new Error(`${TAG} pool destroyed`);
    while (this.waiters.length > 0) {
      const w = this.waiters.shift();
      w?.reject(err);
    }
  }
}
