/**
 * SerialQueue: a lightweight task queue with concurrency=1.
 *
 * Equivalent to `new PQueue({ concurrency: 1 })` but with zero external
 * dependencies. Supports:
 * - Serial execution (FIFO)
 * - `add(fn)` to enqueue a task (returns the task's result promise)
 * - `onIdle()` to wait until all queued tasks have completed
 * - `pause()` / `start()` to suspend/resume execution
 * - `size` to check pending task count
 * - Optional debug logger for enqueue/dequeue/complete diagnostics
 */

type Task<T = unknown> = () => Promise<T>;

interface QueueEntry {
  task: Task;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}

export class SerialQueue {
  /** Human-readable name for logging / diagnostics. */
  public readonly name: string;

  private queue: QueueEntry[] = [];
  private running = false;
  private paused = false;
  private idleResolvers: Array<() => void> = [];

  /** Optional debug logger — receives diagnostic messages for enqueue/dequeue/complete. */
  private debugFn?: (msg: string) => void;

  constructor(name = "unnamed") {
    this.name = name;
  }

  /** Set a debug logger for queue diagnostics. */
  setDebugLogger(fn: (msg: string) => void): void {
    this.debugFn = fn;
  }

  /** Number of tasks waiting to be executed. */
  get size(): number {
    return this.queue.length;
  }

  /** Whether a task is currently executing. */
  get pending(): boolean {
    return this.running;
  }

  /** Whether the queue is idle (no queued tasks and nothing running). */
  get idle(): boolean {
    return this.queue.length === 0 && !this.running;
  }

  /** Add a task to the queue. Returns the task's result promise. */
  add<T>(task: Task<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({
        task: task as Task,
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      this.debugFn?.(`[queue:${this.name}] enqueued, pending=${this.queue.length}, running=${this.running}`);
      this.drain();
    });
  }

  /** Pause the queue. Currently running task will finish, but no new tasks start. */
  pause(): void {
    this.paused = true;
  }

  /** Resume the queue after pause(). */
  start(): void {
    this.paused = false;
    this.drain();
  }

  /** Returns a promise that resolves when all queued tasks have completed. */
  onIdle(): Promise<void> {
    if (this.queue.length === 0 && !this.running) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.idleResolvers.push(resolve);
    });
  }

  /** Clear all pending (not yet started) tasks. */
  clear(): void {
    for (const entry of this.queue) {
      entry.reject(new Error("Queue cleared"));
    }
    this.queue = [];
  }

  private drain(): void {
    if (this.running || this.paused || this.queue.length === 0) return;

    const entry = this.queue.shift()!;
    this.running = true;

    this.debugFn?.(`[queue:${this.name}] dequeued, starting execution (remaining=${this.queue.length})`);

    entry
      .task()
      .then((result) => entry.resolve(result))
      .catch((err) => entry.reject(err))
      .finally(() => {
        this.running = false;
        this.debugFn?.(`[queue:${this.name}] task completed (remaining=${this.queue.length})`);
        if (this.queue.length === 0) {
          // Notify idle waiters
          const resolvers = this.idleResolvers;
          this.idleResolvers = [];
          for (const resolve of resolvers) resolve();
        } else {
          this.drain();
        }
      });
  }
}
