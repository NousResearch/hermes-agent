export interface SourceRuntimeSettleCommand {
  launcher: string
  command: string
  args: string[]
  windowsVerbatimArguments: boolean
}

export function sourceRuntimeSettleCommand(
  root: string,
  env: NodeJS.ProcessEnv | Record<string, string>,
  platform?: NodeJS.Platform,
): SourceRuntimeSettleCommand
