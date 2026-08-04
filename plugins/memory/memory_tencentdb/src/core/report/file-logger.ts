/**
 * file-logger.ts — 本地日志文件写入器
 *
 * 将日志双写到本地文件，支持：
 * - 文件轮转（rotate）
 * - 备份数量限制
 * - 目录自动创建
 * - 错误静默处理（不影响主业务流程）
 *
 * 使用同步文件写入（appendFileSync），避免 WriteStream 导致进程无法退出。
 * 日志写入量小且不频繁，同步开销可忽略。
 */

import fs from "node:fs";
import path from "node:path";

export interface FileLoggerConfig {
  /** 日志文件目录，如 /data/log/。为空时禁用文件写入。 */
  path: string;
  /** 日志文件名，如 core.log */
  filename: string;
  /** 单文件最大字节数，超出后轮转 */
  rotateSizeBytes: number;
  /** 保留的备份文件数量 */
  rotateBackupLimit: number;
}

/**
 * FileLogger 本地日志文件写入器。
 * 实现日志双写到本地文件，支持 rotate。
 */
export class FileLogger {
  private cfg: FileLoggerConfig;
  private currentSize = 0;
  private disabled = false;
  private filePath = "";

  constructor(cfg: FileLoggerConfig) {
    this.cfg = cfg;

    // Path 为空时禁用
    if (!cfg.path) {
      this.disabled = true;
      return;
    }

    // 设置默认值
    if (this.cfg.rotateSizeBytes <= 0) {
      this.cfg.rotateSizeBytes = 100 * 1024 * 1024; // 100MB
    }
    if (this.cfg.rotateBackupLimit <= 0) {
      this.cfg.rotateBackupLimit = 10;
    }

    // 尝试初始化
    try {
      this.initFile();
    } catch (err) {
      this.disabled = true;
      process.stderr.write(`[file-logger] failed to init log file: ${err}\n`);
    }
  }

  /**
   * write 写入一条日志。
   * 格式：[时间][级别] 消息 {json数据}\n
   */
  write(level: string, message: string, data?: Record<string, unknown>): void {
    if (this.disabled) {
      return;
    }

    try {
      const line = this.formatLine(level, message, data);
      this.writeLine(line);
    } catch {
      // 写入失败静默处理，不影响主流程
    }
  }

  /**
   * flush 刷新缓冲区到磁盘（同步写入模式下为 no-op，保留接口兼容）。
   */
  async flush(): Promise<void> {
    // 同步写入模式下，数据已经写入磁盘，无需额外操作
  }

  /**
   * close 关闭（同步写入模式下为 no-op，保留接口兼容）。
   */
  close(): void {
    // 同步写入模式下，无需关闭操作
  }

  // ─── 私有方法 ───

  private formatLine(level: string, message: string, data?: Record<string, unknown>): string {
    const timestamp = new Date().toISOString();
    let line = `[${timestamp}][${level}] ${message}`;

    if (data && Object.keys(data).length > 0) {
      // 按 key 排序以保证输出稳定
      const sorted: Record<string, unknown> = {};
      for (const key of Object.keys(data).sort()) {
        sorted[key] = data[key];
      }
      line += ` ${JSON.stringify(sorted)}`;
    }

    line += "\n";
    return line;
  }

  private writeLine(line: string): void {
    const lineBytes = Buffer.byteLength(line, "utf-8");

    // 检查是否需要轮转
    if (this.currentSize + lineBytes > this.cfg.rotateSizeBytes) {
      this.rotate();
    }

    // 同步追加写入
    fs.appendFileSync(this.filePath, line, "utf-8");
    this.currentSize += lineBytes;
  }

  private initFile(): void {
    // 自动创建目录
    fs.mkdirSync(this.cfg.path, { recursive: true });

    this.filePath = path.join(this.cfg.path, this.cfg.filename);

    // 获取当前文件大小
    try {
      const stat = fs.statSync(this.filePath);
      this.currentSize = stat.size;
    } catch {
      this.currentSize = 0;
    }
  }

  private rotate(): void {
    try {
      this.currentSize = 0;

      // 重命名当前文件为备份（带时间戳）
      const now = new Date();
      const ts = now.toISOString().replace(/[:.]/g, "").replace("T", "_").replace("Z", "");
      const backupName = `${this.cfg.filename}.${ts}`;
      const backupPath = path.join(this.cfg.path, backupName);

      try {
        fs.renameSync(this.filePath, backupPath);
      } catch {
        // rename 失败不影响后续
      }

      // 清理超出 limit 的旧备份
      this.cleanOldBackups();
    } catch (err) {
      this.disabled = true;
      process.stderr.write(`[file-logger] failed to rotate: ${err}\n`);
    }
  }

  private cleanOldBackups(): void {
    try {
      const entries = fs.readdirSync(this.cfg.path);
      const prefix = this.cfg.filename + ".";

      const backups = entries
        .filter((name) => name.startsWith(prefix) && name !== this.cfg.filename)
        .sort(); // 时间戳在后缀，字典序即时间序

      if (backups.length > this.cfg.rotateBackupLimit) {
        const toDelete = backups.slice(0, backups.length - this.cfg.rotateBackupLimit);
        for (const name of toDelete) {
          try {
            fs.unlinkSync(path.join(this.cfg.path, name));
          } catch {
            // 删除失败静默处理
          }
        }
      }
    } catch {
      // 静默处理
    }
  }
}
