import { useQuery } from '@tanstack/react-query'
import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { $desktopLanguage } from '@/store/language'
import type { ModelOptionProvider, ModelOptionsResponse } from '@/types/hermes'

import type { HermesGateway } from '../hermes'
import { getGlobalModelOptions } from '../hermes'
import { cn } from '../lib/utils'

import { InlineNotice } from './notifications'
import { Button } from './ui/button'
import { Checkbox } from './ui/checkbox'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './ui/command'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { Skeleton } from './ui/skeleton'

const pickerPanelClass = 'max-h-[85vh] max-w-2xl gap-0 overflow-hidden p-0'

interface ModelPickerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  gw?: HermesGateway
  sessionId?: string | null
  currentModel: string
  currentProvider: string
  onSelect: (selection: { provider: string; model: string; persistGlobal: boolean }) => void
}

export function ModelPickerDialog({
  open,
  onOpenChange,
  gw,
  sessionId,
  currentModel,
  currentProvider,
  onSelect
}: ModelPickerDialogProps) {
  const language = useStore($desktopLanguage)
  const [persistGlobal, setPersistGlobal] = useState(!sessionId)

  const modelOptions = useQuery({
    queryKey: ['model-options', sessionId || 'global'],
    queryFn: () => {
      if (gw && sessionId) {
        return gw.request<ModelOptionsResponse>('model.options', {
          session_id: sessionId
        })
      }

      return getGlobalModelOptions()
    },
    enabled: open
  })

  const providers = modelOptions.data?.providers ?? []
  const optionsModel = String(modelOptions.data?.model ?? currentModel ?? '')
  const optionsProvider = String(modelOptions.data?.provider ?? currentProvider ?? '')
  const loading = modelOptions.isPending && !modelOptions.data

  const error = modelOptions.error
    ? modelOptions.error instanceof Error
      ? modelOptions.error.message
      : String(modelOptions.error)
    : null

  const selectModel = (provider: ModelOptionProvider, model: string) => {
    onSelect({
      provider: provider.slug,
      model,
      persistGlobal: persistGlobal || !sessionId
    })
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className={pickerPanelClass}>
        <DialogHeader className="border-b border-border px-4 py-3">
          <DialogTitle>{language === 'zh' ? '切换模型' : 'Switch model'}</DialogTitle>
          <DialogDescription className="font-mono text-xs leading-relaxed">
            {language === 'zh' ? '当前' : 'current'}: {optionsModel || currentModel || (language === 'zh' ? '(未知)' : '(unknown)')}
            {optionsProvider || currentProvider ? ` · ${optionsProvider || currentProvider}` : ''}
          </DialogDescription>
        </DialogHeader>

        <Command className="rounded-none bg-card">
          <CommandInput autoFocus placeholder={language === 'zh' ? '筛选提供商和模型...' : 'Filter providers and models...'} />
          <CommandList className="max-h-96">
            {!loading && !error && <CommandEmpty>{language === 'zh' ? '未找到模型。' : 'No models found.'}</CommandEmpty>}
            <ModelResults
              currentModel={optionsModel || currentModel}
              currentProvider={optionsProvider || currentProvider}
              error={error}
              language={language}
              loading={loading}
              onSelectModel={selectModel}
              providers={providers}
            />
          </CommandList>
        </Command>

        <DialogFooter className="flex-row items-center justify-between gap-3 border-t border-border bg-card p-3 sm:justify-between">
          <label className="flex cursor-pointer select-none items-center gap-2 text-xs text-muted-foreground">
            <Checkbox
              checked={persistGlobal || !sessionId}
              disabled={!sessionId}
              onCheckedChange={checked => setPersistGlobal(checked === true)}
            />
            {language === 'zh'
              ? sessionId
                ? '全局保存（否则仅当前会话生效）'
                : '全局保存'
              : sessionId
                ? 'Persist globally (otherwise this session only)'
                : 'Persist globally'}
          </label>

          <Button onClick={() => onOpenChange(false)} variant="outline">
            {language === 'zh' ? '取消' : 'Cancel'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ModelResults({
  loading,
  error,
  providers,
  currentModel,
  currentProvider,
  language,
  onSelectModel
}: {
  loading: boolean
  error: string | null
  providers: ModelOptionProvider[]
  currentModel: string
  currentProvider: string
  language: 'zh' | 'en'
  onSelectModel: (provider: ModelOptionProvider, model: string) => void
}) {
  if (loading) {
    return <LoadingResults />
  }

  if (error) {
    return (
      <div className="px-3 py-3">
        <InlineNotice kind="error" title={language === 'zh' ? '无法加载模型' : 'Could not load models'}>
          {error}
        </InlineNotice>
      </div>
    )
  }

  if (providers.length === 0) {
    return (
      <div className="px-4 py-6 text-sm text-muted-foreground">
        {language === 'zh' ? '没有已认证的模型提供商。' : 'No authenticated providers.'}
      </div>
    )
  }

  return (
    <>
      {providers.map(provider => {
        const models = provider.models ?? []

        if (models.length === 0) {
          return null
        }

        return (
          <CommandGroup heading={<ProviderHeading provider={provider} />} key={provider.slug}>
            {provider.warning && (
              <div className="px-2 pb-2">
                <InlineNotice className="px-2.5 py-1.5 text-xs" kind="warning">
                  {provider.warning}
                </InlineNotice>
              </div>
            )}
            {models.map(model => {
              const isCurrent = model === currentModel && provider.slug === currentProvider

              return (
                <CommandItem
                  className={cn(
                    'pl-6 font-mono',
                    isCurrent &&
                      'bg-primary text-primary-foreground data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground'
                  )}
                  key={`${provider.slug}:${model}`}
                  onSelect={() => onSelectModel(provider, model)}
                  value={`${provider.name} ${provider.slug} ${model}`}
                >
                  <span className="min-w-0 flex-1 truncate">{model}</span>
                </CommandItem>
              )
            })}
          </CommandGroup>
        )
      })}
    </>
  )
}

function LoadingResults() {
  return (
    <CommandGroup heading={<Skeleton className="h-3 w-32" />}>
      {Array.from({ length: 4 }, (_, rowIndex) => (
        <div className="rounded-sm py-1.5 pl-6 pr-2" key={rowIndex}>
          <Skeleton className={cn('h-5', rowIndex % 3 === 0 ? 'w-3/5' : rowIndex % 3 === 1 ? 'w-4/5' : 'w-1/2')} />
        </div>
      ))}
    </CommandGroup>
  )
}

function ProviderHeading({ provider }: { provider: ModelOptionProvider }) {
  return (
    <span className="flex min-w-0 items-center gap-2">
      <span className="truncate">{provider.name}</span>
      <span className="font-mono text-xs font-normal normal-case tracking-normal text-muted-foreground">
        {provider.slug} · {provider.total_models ?? provider.models?.length ?? 0}
      </span>
    </span>
  )
}
