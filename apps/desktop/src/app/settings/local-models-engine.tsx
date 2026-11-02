import { useQuery } from '@tanstack/react-query'
import { type ReactElement, type ReactNode, useState } from 'react'

import { getOmlxStatus, selectLocalEngine } from '@/api/omlx'
import { Button } from '@/components/ui/button'
import { Cpu } from '@/lib/icons'
import {
  isCurrentLocalModelsOwner,
  localModelsKey,
  localModelsRequestScope,
  refreshLocalModels
} from '@/store/local-runtime-jobs'
import { notifyError } from '@/store/notifications'

import { useLocalModelsActionScope } from './local-models-actions'
import { OmlxRuntimeSection } from './local-models-omlx'
import { ListRow, SettingsContent, SettingsSection } from './primitives'

export function LocalModelsEngine({ children }: { children: ReactNode }): ReactElement {
  const { owner, client, copy } = useLocalModelsActionScope()
  const [busy, setBusy] = useState(false)

  const { data: status, error } = useQuery({
    queryKey: localModelsKey(owner, 'omlx'),
    queryFn: () => getOmlxStatus(localModelsRequestScope(owner)),
    refetchInterval: 2_000,
    retry: false
  })

  async function select(engine: 'llamacpp' | 'omlx'): Promise<void> {
    setBusy(true)

    try {
      await selectLocalEngine(engine, localModelsRequestScope(owner))
      await client.invalidateQueries({ queryKey: localModelsKey(owner, 'omlx') })
      refreshLocalModels(owner, client)
    } catch (err) {
      if (isCurrentLocalModelsOwner(owner)) {
        notifyError(err, copy.engineSwitchFailed)
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <SettingsContent>
      {status?.supported && (
        <SettingsSection icon={Cpu} title={copy.runtimeTitle}>
          <ListRow
            action={
              <div aria-label={copy.engineChoice} className="flex gap-2" role="group">
                <Button
                  aria-pressed={!status?.selected}
                  disabled={!status || busy}
                  onClick={() => void select('llamacpp')}
                  size="sm"
                  variant={status?.selected ? 'outline' : 'default'}
                >
                  llama.cpp
                </Button>
                <Button
                  aria-pressed={status?.selected ?? false}
                  disabled={!status || !status.supported || busy}
                  onClick={() => void select('omlx')}
                  size="sm"
                  variant={status?.selected ? 'default' : 'outline'}
                >
                  oMLX
                </Button>
              </div>
            }
            description={copy.engineChoiceDetail}
            title={copy.engineChoice}
          />

          {error && (
            <p className="text-sm text-destructive" role="alert">
              {copy.engineStatusFailed}
            </p>
          )}
        </SettingsSection>
      )}
      {status?.supported && status.selected ? <OmlxRuntimeSection status={status} /> : children}
    </SettingsContent>
  )
}
