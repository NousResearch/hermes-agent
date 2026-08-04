/**
 * 元数据 API trace → stdout 单行 JSON（CLS 复用）。
 */
export const API_TRACE_INTERFACE = "tdai-metadata-api";

type StdoutWriter = (line: string) => void;

let stdoutWriter: StdoutWriter = (line) => {
  process.stdout.write(line);
};

export function buildStdoutPayload(
  level: string,
  event: string,
  profile: string,
  merged: Record<string, string | number | boolean>,
): Record<string, string | number | boolean> {
  return {
    interface: API_TRACE_INTERFACE,
    time: new Date().toISOString(),
    level: level.toUpperCase(),
    msg: event,
    profile,
    ...merged,
  };
}

export function writeApiTraceStdout(payload: Record<string, unknown>): void {
  try {
    const line = `${JSON.stringify(payload)}\n`;
    stdoutWriter(line);
  } catch {
    // 静默失败
  }
}

/** 测试用：替换 stdout 写入器。 */
export function setStdoutWriterForTests(writer: StdoutWriter | null): void {
  stdoutWriter = writer ?? ((line) => process.stdout.write(line));
}

/** 测试用：读取当前 stdout 写入器。 */
export function getStdoutWriterForTests(): StdoutWriter {
  return stdoutWriter;
}
