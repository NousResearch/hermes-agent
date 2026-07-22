export function shouldBlockUpdateClose(
  mode: 'install' | 'update',
  status: 'completed' | 'failed' | 'idle' | 'running'
): boolean {
  return mode === 'update' && status === 'running'
}
