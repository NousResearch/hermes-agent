type MetroStackFrame = {
  column?: number | string | null;
  file?: string | null;
  lineNumber?: number | string | null;
  methodName?: string | null;
};

export type SymbolicatedStackTrace = {
  stack: MetroStackFrame[];
  codeFrame?: unknown;
};

export function getBaseUrl() {
  return 'http://localhost:8081';
}

export function openFileInEditor() {
  // No-op in the embedded/static mobile app. There is no Metro editor endpoint.
}

export async function fetchProjectMetadataAsync() {
  return { projectRoot: '', serverRoot: '', sdkVersion: '' };
}

export async function symbolicateStackTrace(stack: MetroStackFrame[]): Promise<SymbolicatedStackTrace> {
  return { stack };
}

export function formatProjectFilePath(_projectRoot = '', file: string | null = null): string {
  return file || '<unknown>';
}

export function getStackFormattedLocation(_projectRoot: string | undefined, frame: MetroStackFrame): string {
  const file = formatProjectFilePath('', frame.file ?? null);
  const line = frame.lineNumber ?? 0;
  const column = frame.column ?? 0;
  return `${file}:${line}:${column}`;
}

export function getFormattedStackTrace(stack: MetroStackFrame[], projectRoot = '') {
  return stack
    .map((frame) => `  at ${frame.methodName ?? '<unknown>'} (${getStackFormattedLocation(projectRoot, frame)})`)
    .join('\n');
}

export function isStackFileAnonymous(frame: MetroStackFrame): boolean {
  return !frame.file || frame.file === '<unknown>' || frame.file === '<anonymous>';
}
