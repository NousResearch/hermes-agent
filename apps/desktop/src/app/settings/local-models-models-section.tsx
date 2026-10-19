import type { ReactElement } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Download, Search } from '@/lib/icons'
import type { LocalCatalogModel, LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

import { CatalogModelRow, SideloadedModelRow } from './local-models-model-rows'
import { ListRow, SettingsSection } from './primitives'

// Catalog display order: what runs well leads. Resident (all on GPU)
// first, then spilled (works, slower), then doesn't-fit; catalog order
// (recommended first) holds within each band.
function fitRank(model: LocalCatalogModel): number {
  if (model.fits && !model.spilled) {
    return 0
  }

  if (model.fits) {
    return 1
  }

  return 2
}

export interface LocalModelsModelsSectionProps {
  status: LocalModelsStatus
  catalog: LocalCatalogModel[]
  jobs: readonly LocalRuntimeJob[]
  lastError: LocalRuntimeJob | undefined
}

export function LocalModelsModelsSection({
  status,
  catalog,
  jobs,
  lastError
}: LocalModelsModelsSectionProps): ReactElement {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const hasRecommendation = catalog.some(c => c.recommended)
  const sortedCatalog = [...catalog].sort((a, b) => fitRank(a) - fitRank(b))

  return (
    <SettingsSection icon={Download} meta={`${catalog.length}`} title={copy.modelsTitle}>
      {!hasRecommendation && (
        <ListRow
          action={
            <Button
              onClick={() =>
                document.getElementById('local-model-browse')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
              }
              size="sm"
            >
              <Search />
              {copy.noRecommendationAction}
            </Button>
          }
          description={copy.noRecommendationDetail}
          title={copy.noRecommendationTitle}
        />
      )}

      <div className="grid gap-1">
        {sortedCatalog.map(model => (
          <CatalogModelRow jobs={jobs} key={model.id} model={model} status={status} />
        ))}

        {status.models
          .filter(m => !catalog.some(c => c.downloaded_model_id === m.id || c.model_id === m.id))
          .map(m => (
            <SideloadedModelRow jobs={jobs} key={m.id} model={m} status={status} />
          ))}
      </div>

      {lastError?.kind === 'model-download' && <p className="text-[0.75rem] text-destructive">{lastError.error}</p>}
    </SettingsSection>
  )
}
