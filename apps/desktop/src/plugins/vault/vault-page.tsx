import {
  Badge,
  Button,
  Codicon,
  Input,
  Loader,
  ScrollArea,
  Streamdown,
  Textarea,
  Tip,
  useMutation,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { useMemo, useState } from 'react'

import {
  $selectedNoteTitle,
  deleteVaultNote,
  fetchVaultGraph,
  fetchVaultNote,
  fetchVaultNotes,
  saveVaultNote
} from './api'
import { VaultGraphView } from './graph-view'
import type { VaultNote, VaultNoteSummary } from './types'

export function VaultPage() {
  const qc = useQueryClient()
  const selectedTitle = useValue($selectedNoteTitle)

  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTag, setSelectedTag] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'editor' | 'preview' | 'graph'>('editor')

  // Local draft state for editing
  const [draftTitle, setDraftTitle] = useState('')
  const [draftContent, setDraftContent] = useState('')
  const [draftTags, setDraftTags] = useState('')
  const [isCreatingNew, setIsCreatingNew] = useState(false)

  // 1. Fetch notes list
  const { data: notesData, isLoading: isNotesLoading } = useQuery({
    queryKey: ['vault', 'notes'],
    queryFn: fetchVaultNotes,
    refetchInterval: 10_000
  })
  const notes = notesData?.notes ?? []

  // Auto-select first note if none selected
  const activeTitle = selectedTitle || (notes[0]?.title ?? '')

  // 2. Fetch active note details
  const { data: noteData, isLoading: isNoteLoading } = useQuery({
    queryKey: ['vault', 'note', activeTitle],
    queryFn: () => fetchVaultNote(activeTitle),
    enabled: Boolean(activeTitle) && !isCreatingNew
  })
  const activeNote = noteData?.note

  // Synchronize draft state when active note changes
  useMemo(() => {
    if (activeNote && !isCreatingNew) {
      setDraftTitle(activeNote.title)
      setDraftContent(activeNote.content)
      setDraftTags(activeNote.tags.join(', '))
    }
  }, [activeNote, isCreatingNew])

  // 3. Fetch Graph
  const { data: graphData } = useQuery({
    queryKey: ['vault', 'graph'],
    queryFn: fetchVaultGraph,
    enabled: viewMode === 'graph'
  })
  const graph = graphData?.graph ?? { nodes: [], edges: [] }

  // 4. Mutations
  const saveMutation = useMutation({
    mutationFn: async () => {
      const tags = draftTags
        .split(',')
        .map(t => t.trim().replace(/^#/, ''))
        .filter(Boolean)
      const res = await saveVaultNote({
        title: draftTitle.trim() || 'Sem Título',
        content: draftContent,
        tags
      })
      return res
    },
    onSuccess: res => {
      setIsCreatingNew(false)
      $selectedNoteTitle.set(res.note.title)
      void qc.invalidateQueries({ queryKey: ['vault'] })
    }
  })

  const deleteMutation = useMutation({
    mutationFn: async (title: string) => {
      await deleteVaultNote(title)
    },
    onSuccess: () => {
      $selectedNoteTitle.set('')
      void qc.invalidateQueries({ queryKey: ['vault'] })
    }
  })

  // Filter notes
  const filteredNotes = useMemo(() => {
    return notes.filter(n => {
      const matchSearch =
        !searchQuery ||
        n.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        n.tags.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()))
      const matchTag = !selectedTag || n.tags.includes(selectedTag)
      return matchSearch && matchTag
    })
  }, [notes, searchQuery, selectedTag])

  // Extract all unique tags
  const allTags = useMemo(() => {
    const s = new Set<string>()
    for (const n of notes) {
      for (const t of n.tags) s.add(t)
    }
    return Array.from(s).sort()
  }, [notes])

  const handleStartNewNote = () => {
    setIsCreatingNew(true)
    setDraftTitle('Nova Nota')
    setDraftContent('# Nova Nota\n\nComece a escrever e use [[Wikilinks]] para conectar ideias.')
    setDraftTags('')
    setViewMode('editor')
  }

  const handleSelectNote = (title: string) => {
    setIsCreatingNew(false)
    $selectedNoteTitle.set(title)
  }

  return (
    <div className="flex h-full w-full overflow-hidden bg-(--ui-bg-primary) text-xs">
      {/* 1. Left Rail: Explorer */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-(--ui-stroke-secondary) bg-(--ui-bg-secondary)">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-(--ui-stroke-secondary) p-3">
          <div className="flex items-center gap-1.5 font-semibold text-(--ui-text-primary)">
            <Codicon name="book" size="1rem" />
            <span>Hermes Vault</span>
            <Badge className="ml-1 text-[10px]" variant="muted">
              {notes.length}
            </Badge>
          </div>
          <Button onClick={handleStartNewNote} size="xs">
            <Codicon name="add" size="0.875rem" /> Nova
          </Button>
        </div>

        {/* Search */}
        <div className="p-2">
          <Input
            aria-label="Buscar notas..."
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Buscar notas..."
            value={searchQuery}
          />
        </div>

        {/* Tags horizontal scroll */}
        {allTags.length > 0 && (
          <div className="flex gap-1 overflow-x-auto px-2 pb-2 text-[10px]">
            <button
              className={`rounded px-1.5 py-0.5 ${!selectedTag ? 'bg-(--ui-accent-primary) text-white' : 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'}`}
              onClick={() => setSelectedTag(null)}
              type="button"
            >
              Todas
            </button>
            {allTags.map(tag => (
              <button
                className={`rounded px-1.5 py-0.5 ${selectedTag === tag ? 'bg-(--ui-accent-primary) text-white' : 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'}`}
                key={tag}
                onClick={() => setSelectedTag(tag === selectedTag ? null : tag)}
                type="button"
              >
                #{tag}
              </button>
            ))}
          </div>
        )}

        {/* Notes List */}
        <ScrollArea className="flex-1 p-2">
          {isNotesLoading ? (
            <div className="grid place-items-center py-8">
              <Loader type="lemniscate-bloom" />
            </div>
          ) : filteredNotes.length === 0 ? (
            <div className="py-8 text-center text-(--ui-text-tertiary)">Nenhuma nota encontrada.</div>
          ) : (
            <div className="flex flex-col gap-1">
              {filteredNotes.map(n => {
                const isSelected = !isCreatingNew && n.title === activeTitle
                return (
                  <button
                    className={`flex flex-col items-start gap-1 rounded-md p-2 text-left transition-colors ${
                      isSelected
                        ? 'bg-(--ui-bg-quaternary) text-(--ui-text-primary) font-medium shadow-xs'
                        : 'hover:bg-(--ui-bg-tertiary) text-(--ui-text-secondary)'
                    }`}
                    key={n.title}
                    onClick={() => handleSelectNote(n.title)}
                    type="button"
                  >
                    <div className="flex w-full items-center justify-between">
                      <span className="truncate">{n.title}</span>
                      {n.backlinks_count > 0 && (
                        <span className="flex items-center gap-0.5 text-[10px] text-(--ui-text-tertiary)">
                          <Codicon name="link" size="0.75rem" /> {n.backlinks_count}
                        </span>
                      )}
                    </div>
                    {n.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {n.tags.slice(0, 3).map(t => (
                          <span
                            className="rounded bg-(--ui-bg-primary) px-1 py-0.2 text-[9px] text-(--ui-text-tertiary)"
                            key={t}
                          >
                            #{t}
                          </span>
                        ))}
                      </div>
                    )}
                  </button>
                )
              })}
            </div>
          )}
        </ScrollArea>
      </aside>

      {/* 2. Center: Note Editor & Knowledge Graph */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Sub-header Toolbar */}
        <header className="flex items-center justify-between border-b border-(--ui-stroke-secondary) px-4 py-2 bg-(--ui-bg-primary)">
          <div className="flex items-center gap-2">
            <div className="flex rounded bg-(--ui-bg-tertiary) p-0.5">
              <button
                className={`rounded px-2.5 py-1 text-xs font-medium transition-all ${
                  viewMode === 'editor' ? 'bg-(--ui-bg-primary) text-(--ui-text-primary) shadow-xs' : 'text-(--ui-text-tertiary)'
                }`}
                onClick={() => setViewMode('editor')}
                type="button"
              >
                Editor
              </button>
              <button
                className={`rounded px-2.5 py-1 text-xs font-medium transition-all ${
                  viewMode === 'preview' ? 'bg-(--ui-bg-primary) text-(--ui-text-primary) shadow-xs' : 'text-(--ui-text-tertiary)'
                }`}
                onClick={() => setViewMode('preview')}
                type="button"
              >
                Preview
              </button>
              <button
                className={`rounded px-2.5 py-1 text-xs font-medium transition-all ${
                  viewMode === 'graph' ? 'bg-(--ui-bg-primary) text-(--ui-text-primary) shadow-xs' : 'text-(--ui-text-tertiary)'
                }`}
                onClick={() => setViewMode('graph')}
                type="button"
              >
                Grafo
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {viewMode !== 'graph' && (
              <>
                <Button
                  disabled={saveMutation.isPending}
                  onClick={() => void saveMutation.mutate()}
                  size="xs"
                >
                  <Codicon name="save" size="0.875rem" /> {saveMutation.isPending ? 'Salvando...' : 'Salvar'}
                </Button>
                {!isCreatingNew && activeTitle && (
                  <Button
                    onClick={() => {
                      if (confirm(`Deseja realmente excluir "${activeTitle}"?`)) {
                        void deleteMutation.mutate(activeTitle)
                      }
                    }}
                    size="xs"
                    variant="ghost"
                  >
                    <Codicon name="trash" size="0.875rem" />
                  </Button>
                )}
              </>
            )}
          </div>
        </header>

        {/* Content Body */}
        {viewMode === 'graph' ? (
          <div className="flex-1 overflow-hidden">
            <VaultGraphView graph={graph} onSelectNote={handleSelectNote} />
          </div>
        ) : isNoteLoading && !isCreatingNew ? (
          <div className="grid flex-1 place-items-center">
            <Loader type="lemniscate-bloom" />
          </div>
        ) : !activeNote && !isCreatingNew ? (
          <div className="grid flex-1 place-items-center text-sm text-(--ui-text-tertiary)">
            Selecione uma nota ou crie uma nova para começar.
          </div>
        ) : (
          <div className="flex flex-1 flex-col overflow-hidden p-4">
            {/* Note Title & Tags Editor */}
            <div className="mb-3 flex flex-col gap-2">
              <Input
                aria-label="Título da nota"
                className="text-base font-bold"
                onChange={e => setDraftTitle(e.target.value)}
                placeholder="Título da nota..."
                value={draftTitle}
              />
              <Input
                aria-label="Tags (separadas por vírgula)"
                className="text-xs"
                onChange={e => setDraftTags(e.target.value)}
                placeholder="Tags (separadas por vírgula, ex: projeto, pesquisa)..."
                value={draftTags}
              />
            </div>

            {/* Note Body */}
            {viewMode === 'preview' ? (
              <ScrollArea className="flex-1 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) p-4">
                <Streamdown controls={false} mode="static">{draftContent}</Streamdown>
              </ScrollArea>
            ) : (
              <Textarea
                aria-label="Conteúdo Markdown"
                className="flex-1 resize-none font-mono text-xs leading-relaxed"
                onChange={e => setDraftContent(e.target.value)}
                placeholder="Escreva sua nota em Markdown. Use [[Wikilinks]] para conectar ideias..."
                value={draftContent}
              />
            )}
          </div>
        )}
      </main>

      {/* 3. Right Rail: Inspector & Backlinks */}
      {viewMode !== 'graph' && activeNote && !isCreatingNew && (
        <aside className="flex w-60 shrink-0 flex-col border-l border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
          <div className="mb-3 font-semibold text-(--ui-text-primary)">Conexões & Metadados</div>

          {/* Linked Mentions (Backlinks) */}
          <div className="mb-4">
            <div className="mb-1.5 flex items-center justify-between text-[11px] font-medium text-(--ui-text-secondary)">
              <span>Mencionado em ({activeNote.backlinks.length})</span>
            </div>
            {activeNote.backlinks.length === 0 ? (
              <div className="text-[10px] text-(--ui-text-tertiary)">Nenhum backlink ainda.</div>
            ) : (
              <div className="flex flex-col gap-1">
                {activeNote.backlinks.map(source => (
                  <button
                    className="rounded bg-(--ui-bg-primary) p-1.5 text-left text-xs font-medium hover:border-(--ui-stroke-primary) border border-(--ui-stroke-secondary)"
                    key={source}
                    onClick={() => handleSelectNote(source)}
                    type="button"
                  >
                    [[{source}]]
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Outgoing Wikilinks */}
          <div className="mb-4">
            <div className="mb-1.5 flex items-center justify-between text-[11px] font-medium text-(--ui-text-secondary)">
              <span>Links de Saída ({activeNote.forward_links.length})</span>
            </div>
            {activeNote.forward_links.length === 0 ? (
              <div className="text-[10px] text-(--ui-text-tertiary)">Nenhum link de saída.</div>
            ) : (
              <div className="flex flex-col gap-1">
                {activeNote.forward_links.map(target => (
                  <button
                    className="rounded bg-(--ui-bg-primary) p-1.5 text-left text-xs text-(--ui-text-secondary) hover:border-(--ui-stroke-primary) border border-(--ui-stroke-secondary)"
                    key={target}
                    onClick={() => handleSelectNote(target)}
                    type="button"
                  >
                    [[{target}]]
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Note Metadata Details */}
          <div className="mt-auto border-t border-(--ui-stroke-secondary) pt-3 text-[10px] text-(--ui-text-tertiary)">
            <div>Caminho: {activeNote.rel_path}</div>
            <div>Tamanho: {activeNote.size} bytes</div>
            <div className="mt-2">
              <Button
                className="w-full text-center"
                onClick={() => {
                  navigator.clipboard?.writeText(`Por favor, consulte minha nota do Vault [[${activeNote.title}]]`)
                }}
                size="xs"
                variant="outline"
              >
                Copiar ref p/ Chat
              </Button>
            </div>
          </div>
        </aside>
      )}
    </div>
  )
}
