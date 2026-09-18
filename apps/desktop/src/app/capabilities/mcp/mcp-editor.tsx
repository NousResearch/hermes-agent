// The mcp.json editor, and the editor-with-logs pane that used to be the right
// half of `mcp-tab.tsx`.
//
// Extracted so the Connectors page opens the SAME editor — one document, one
// save path — inside its `Add your own` dialog and inside a local server's
// `Advanced` section. The page passes `highlightServer` to scope the highlight
// to the server the dialog is about; the tab lets the cursor decide.

import { useState } from 'react'

import { JsonDocumentEditor } from '@/components/chat/json-document-editor'
import { Button } from '@/components/ui/button'
import { TextTab } from '@/components/ui/text-tab'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

import { DetailPane } from '../../master-detail'

import { McpLogs, type McpLogSource } from './mcp-logs'
import type { McpServersController } from './use-mcp-servers'

export interface McpJsonEditorProps {
  controller: McpServersController
  /** Highlight THIS server's block instead of the one under the cursor. */
  highlightServer?: null | string
}

export function McpJsonEditor({ controller, highlightServer }: McpJsonEditorProps) {
  const { t } = useI18n()
  const m = t.settings.mcp

  const block =
    highlightServer === undefined
      ? controller.activeBlock
      : (controller.blocks.find(candidate => candidate.name === highlightServer) ?? null)

  return (
    <JsonDocumentEditor
      apiRef={controller.editorApi}
      disabled={controller.saving}
      filePath="mcp.json"
      header={
        <>
          mcp.json
          {controller.dirty && <span aria-hidden className="size-1.5 rounded-full bg-current/60" />}
        </>
      }
      highlight={block ? { from: block.from, to: block.to } : null}
      initialValue={controller.draft}
      onChange={controller.setDraft}
      onCursorChange={controller.setCursor}
      onFormatJsonError={error => notifyError(new Error(error), m.invalidJson)}
      onSave={() => void controller.saveDoc()}
      remountKey={controller.docVersion}
      trailing={
        <Button disabled={controller.saving || !controller.dirty} onClick={() => void controller.saveDoc()} size="xs">
          {controller.saving ? t.common.saving : t.common.save}
        </Button>
      }
    />
  )
}

export interface McpLogPaneProps {
  /** `null` tails every server. */
  server: null | string
}

/** The log pane with its stdio/agent switch, pinned under an editor. */
export function McpLogPane({ server }: McpLogPaneProps) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const [source, setSource] = useState<McpLogSource>('stdio')

  return (
    <DetailPane
      actions={
        <span className="flex items-center gap-1.5">
          {(['stdio', 'agent'] as const).map(kind => (
            <TextTab
              active={source === kind}
              className="h-5 px-0.5 text-[0.65rem]"
              key={kind}
              onClick={() => setSource(kind)}
            >
              {kind}
            </TextTab>
          ))}
        </span>
      }
      defaultHeight={176}
      id="mcp-logs"
      title={<span className="text-[0.68rem] font-normal text-muted-foreground/60">{server ?? m.allServers}</span>}
    >
      <McpLogs emptyLabel={m.noOutput} server={server} source={source} />
    </DetailPane>
  )
}

/** The editor with the logs hard-pinned below it: the MCP tab's right column. */
export function McpEditorPane({ controller }: { controller: McpServersController }) {
  const saved = controller.selected !== null && controller.servers[controller.selected] !== undefined

  return (
    <main className="flex min-h-0 flex-col overflow-hidden">
      <McpJsonEditor controller={controller} />
      <McpLogPane server={saved ? controller.selected : null} />
    </main>
  )
}
