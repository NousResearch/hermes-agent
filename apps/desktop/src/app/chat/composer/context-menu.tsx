import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { composerPanelCard } from '@/components/chat/composer-dock'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Kbd } from '@/components/ui/kbd'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import {
  Clipboard,
  FileText,
  FolderOpen,
  type IconComponent,
  ImageIcon,
  Link,
  MessageSquareText,
  Pencil,
  Plus,
  Trash2
} from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $promptTemplates,
  addTemplate,
  deleteTemplate,
  ensureSeeded,
  moveTemplateDown,
  moveTemplateUp,
  type PromptTemplate,
  resetToBuiltins,
  updateTemplate
} from '@/store/prompt-templates'

import { useComposerAttachmentProviders } from './contrib'
import { GHOST_ICON_BTN } from './controls'
import type { ChatBarState } from './types'

export function ContextMenu({
  state,
  onInsertText,
  onOpenUrlDialog,
  onPasteClipboardImage,
  onPickFiles,
  onPickFolders,
  onPickImages
}: ContextMenuProps) {
  const { t } = useI18n()
  const c = t.composer
  // Prompt templates used to be a Radix submenu. That submenu didn't open
  // reliably when the parent menu was positioned at the bottom of the
  // window (composer "+" anchor), so we promoted it to a real Dialog —
  // easier to grow with search / descriptions, and no positioning math.
  const [templatesOpen, setTemplatesOpen] = useState(false)
  // `composer.attachments` contributions — plugin/core-registered rows that
  // extend this menu through the same registry as every other surface.
  const attachmentProviders = useComposerAttachmentProviders()

  return (
    <>
      <DropdownMenu>
        <Tip label={state.tools.label} side="top">
          <DropdownMenuTrigger asChild>
            <Button
              aria-label={state.tools.label}
              className={cn(
                GHOST_ICON_BTN,
                'data-[state=open]:bg-(--chrome-action-hover) data-[state=open]:text-foreground'
              )}
              disabled={!state.tools.enabled}
              size="icon"
              type="button"
              variant="ghost"
            >
              <Codicon name="add" size="0.875rem" />
            </Button>
          </DropdownMenuTrigger>
        </Tip>
        <DropdownMenuContent align="start" className={cn('w-60', composerPanelCard)} side="top" sideOffset={6}>
          <DropdownMenuLabel className="px-2 pb-0.5 pt-0.5 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
            {c.attachLabel}
          </DropdownMenuLabel>
          <ContextMenuItem disabled={!onPickFiles} icon={FileText} onSelect={onPickFiles}>
            {c.files}
          </ContextMenuItem>
          <ContextMenuItem disabled={!onPickFolders} icon={FolderOpen} onSelect={onPickFolders}>
            {c.folder}
          </ContextMenuItem>
          <ContextMenuItem disabled={!onPickImages} icon={ImageIcon} onSelect={onPickImages}>
            {c.images}
          </ContextMenuItem>
          <ContextMenuItem
            disabled={!onPasteClipboardImage}
            icon={Clipboard}
            onSelect={onPasteClipboardImage ? () => void onPasteClipboardImage() : undefined}
          >
            {c.pasteImage}
          </ContextMenuItem>
          <ContextMenuItem icon={Link} onSelect={onOpenUrlDialog}>
            {c.url}
          </ContextMenuItem>

          <DropdownMenuSeparator />

          <ContextMenuItem icon={MessageSquareText} onSelect={() => setTemplatesOpen(true)}>
            {c.promptTemplates}
          </ContextMenuItem>

          {attachmentProviders.length > 0 && <DropdownMenuSeparator />}
          {attachmentProviders.map(provider => (
            <DropdownMenuItem
              className="text-[length:var(--conversation-tool-font-size)] focus:bg-(--ui-bg-tertiary)"
              key={provider.key}
              onSelect={() => void provider.run({ insertText: onInsertText })}
            >
              <Codicon name={provider.icon ?? 'plug'} size="0.875rem" />
              <span>{provider.label}</span>
            </DropdownMenuItem>
          ))}

          <DropdownMenuSeparator />

          <div className="px-2 py-1 text-[0.7rem] text-muted-foreground/80">
            {c.tipPre}
            <Kbd size="sm">@</Kbd>
            {c.tipPost}
          </div>
        </DropdownMenuContent>
      </DropdownMenu>

      <PromptTemplatesDialog onInsertText={onInsertText} onOpenChange={setTemplatesOpen} open={templatesOpen} />
    </>
  )
}

function PromptTemplatesDialog({ onInsertText, onOpenChange, open }: PromptTemplatesDialogProps) {
  const { t } = useI18n()
  const c = t.composer
  const templates = useStore($promptTemplates)
  const [editingId, setEditingId] = useState<string | null>(null)

  // Seed built-in templates (locale-aware) the first time the dialog opens.
  // Done here rather than at module load so translateNow sees the active
  // locale, not the default — avoids English text on a Chinese UI.
  useEffect(() => {
    if (open) {
      ensureSeeded()
    }
  }, [open])

  // Drop edit state if the template being edited was deleted
  useEffect(() => {
    if (editingId && !templates.some(s => s.id === editingId)) {
      setEditingId(null)
    }
  }, [editingId, templates])

  function handleAdd() {
    const created = addTemplate()
    setEditingId(created.id)
  }

  function handleReset() {
    if (window.confirm(c.templateResetConfirm)) {
      resetToBuiltins()
      setEditingId(null)
    }
  }

  function handleInsert(template: PromptTemplate) {
    onInsertText(template.text)
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{c.templatesTitle}</DialogTitle>
          <DialogDescription>{c.templatesDesc}</DialogDescription>
        </DialogHeader>

        {templates.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">{c.templateEmpty}</p>
        ) : (
          <ul className="grid max-h-[40vh] gap-1 overflow-y-auto">
            {templates.map((template, index) => (
              <li key={template.id}>
                {editingId === template.id ? (
                  <TemplateEditor
                    onCancel={() => setEditingId(null)}
                    onSave={() => setEditingId(null)}
                    template={template}
                  />
                ) : (
                  <TemplateRow
                    canMoveDown={index < templates.length - 1}
                    canMoveUp={index > 0}
                    onDelete={() => {
                      if (window.confirm(c.templateConfirmDelete)) {
                        deleteTemplate(template.id)
                      }
                    }}
                    onEdit={() => setEditingId(template.id)}
                    onInsert={() => handleInsert(template)}
                    onMoveDown={() => moveTemplateDown(template.id)}
                    onMoveUp={() => moveTemplateUp(template.id)}
                    template={template}
                  />
                )}
              </li>
            ))}
          </ul>
        )}

        <div className="flex items-center justify-between border-t pt-3">
          <Button onClick={handleReset} size="sm" variant="ghost">
            {c.templateReset}
          </Button>
          <Button onClick={handleAdd} size="sm" variant="outline">
            <Plus className="size-4" />
            {c.templateAdd}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function TemplateRow({
  canMoveDown,
  canMoveUp,
  onDelete,
  onEdit,
  onInsert,
  onMoveDown,
  onMoveUp,
  template
}: TemplateRowProps) {
  const { t } = useI18n()
  const c = t.composer

  return (
    <div className="group/template flex w-full items-start gap-2.5 rounded-md border border-transparent px-2.5 py-2 text-left transition-colors hover:border-(--ui-stroke-tertiary) hover:bg-(--ui-control-hover-background)">
      <button
        className="grid min-w-0 flex-1 cursor-pointer items-start gap-0.5 text-left"
        onClick={onInsert}
        type="button"
      >
        <span className="text-sm font-medium text-foreground">
          {template.label || <span className="text-muted-foreground italic">{c.templateLabelPlaceholder}</span>}
        </span>
        {template.description && (
          <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {template.description}
          </span>
        )}
        <span className="truncate text-[length:var(--conversation-caption-font-size)] text-muted-foreground/70">
          {template.text}
        </span>
      </button>

      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover/template:opacity-100">
        <Tip label={c.templateMoveUp} side="top">
          <Button disabled={!canMoveUp} onClick={onMoveUp} size="icon" type="button" variant="ghost">
            <Codicon name="chevron-up" size="0.875rem" />
          </Button>
        </Tip>
        <Tip label={c.templateMoveDown} side="top">
          <Button disabled={!canMoveDown} onClick={onMoveDown} size="icon" type="button" variant="ghost">
            <Codicon name="chevron-down" size="0.875rem" />
          </Button>
        </Tip>
        <Tip label={c.templateEdit} side="top">
          <Button onClick={onEdit} size="icon" type="button" variant="ghost">
            <Pencil className="size-3.5" />
          </Button>
        </Tip>
        <Tip label={c.templateDelete} side="top">
          <Button onClick={onDelete} size="icon" type="button" variant="ghost">
            <Trash2 className="size-3.5" />
          </Button>
        </Tip>
      </div>
    </div>
  )
}

function TemplateEditor({ onCancel, onSave, template }: TemplateEditorProps) {
  const { t } = useI18n()
  const c = t.composer
  const [label, setLabel] = useState(template.label)
  const [description, setDescription] = useState(template.description)
  const [text, setText] = useState(template.text)

  // Focus the label field on mount
  const labelRef = (el: HTMLInputElement | null) => {
    el?.focus()
  }

  function handleSave() {
    updateTemplate(template.id, {
      label: label.trim(),
      description: description.trim(),
      text: text.trim()
    })
    onSave()
  }

  return (
    <div className="grid gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary) p-3">
      <Input
        onChange={e => setLabel(e.target.value)}
        placeholder={c.templateLabelPlaceholder}
        ref={labelRef}
        value={label}
      />
      <Input
        onChange={e => setDescription(e.target.value)}
        placeholder={c.templateDescPlaceholder}
        value={description}
      />
      <Textarea onChange={e => setText(e.target.value)} placeholder={c.templateTextPlaceholder} rows={3} value={text} />
      <div className="flex justify-end gap-2">
        <Button onClick={onCancel} size="sm" type="button" variant="ghost">
          {c.templateCancel}
        </Button>
        <Button onClick={handleSave} size="sm" type="button">
          {c.templateSave}
        </Button>
      </div>
    </div>
  )
}

export function ContextMenuItem({ children, disabled, icon: Icon, onSelect }: ContextMenuItemProps) {
  return (
    // Override font size + highlight to match the / · @ completion rows exactly.
    <DropdownMenuItem
      className="text-[length:var(--conversation-tool-font-size)] focus:bg-(--ui-bg-tertiary)"
      disabled={disabled}
      onSelect={onSelect}
    >
      <Icon />
      <span>{children}</span>
    </DropdownMenuItem>
  )
}

interface ContextMenuItemProps {
  children: string
  disabled?: boolean
  icon: IconComponent
  onSelect?: () => void
}

interface ContextMenuProps {
  onInsertText: (text: string) => void
  onOpenUrlDialog: () => void
  onPasteClipboardImage?: (opts?: { silent?: boolean }) => Promise<boolean> | void
  onPickFiles?: () => void
  onPickFolders?: () => void
  onPickImages?: () => void
  state: ChatBarState
}

interface PromptTemplatesDialogProps {
  onInsertText: (text: string) => void
  onOpenChange: (open: boolean) => void
  open: boolean
}

interface TemplateRowProps {
  canMoveDown: boolean
  canMoveUp: boolean
  onDelete: () => void
  onEdit: () => void
  onInsert: () => void
  onMoveDown: () => void
  onMoveUp: () => void
  template: PromptTemplate
}

interface TemplateEditorProps {
  onCancel: () => void
  onSave: () => void
  template: PromptTemplate
}
