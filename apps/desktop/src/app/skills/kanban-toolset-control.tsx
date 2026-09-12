import { useMutation, useQuery } from '@tanstack/react-query'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import {
  getApiRequestConnection,
  getApiRequestProfile,
  getToolsets,
  type ProfileScope,
  profileScopeKey,
  setToolsetEnabled
} from '@/hermes'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { notify, notifyError } from '@/store/notifications'
import type { ToolsetInfo } from '@/types/hermes'

import { SKILLS_QUERY_KEY, TOOLSETS_QUERY_KEY } from './store'

/** The built-in board has no installable agent-plugin half: its tools use the
 * existing per-profile tool configurator. The Desktop switch stays UI-only. */
export function KanbanToolsetControl({ profile, scopeLabel }: { profile: ProfileScope; scopeLabel: string }) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const scopeKey = profileScopeKey(profile)

  const query = useQuery(
    {
      queryKey: [...TOOLSETS_QUERY_KEY, scopeKey],
      queryFn: () => getToolsets(profile),
      retry: false,
      staleTime: 0
    },
    queryClient
  )

  const row = query.data?.find(toolset => toolset.name === 'kanban')

  const mutation = useMutation(
    {
      mutationFn: async (input: { enabled: boolean; profile: ProfileScope; scopeKey: string; label: string }) => {
        const result = await setToolsetEnabled('kanban', input.enabled, input.profile)

        if (!result.ok || result.name !== 'kanban' || result.enabled !== input.enabled) {
          throw new Error(p.toggleFailed('Kanban'))
        }

        return result
      },
      onMutate: async input => {
        const key = [...TOOLSETS_QUERY_KEY, input.scopeKey]
        await queryClient.cancelQueries({ queryKey: key, exact: true })
        const previous = queryClient.getQueryData<ToolsetInfo[]>(key)?.find(item => item.name === 'kanban')
        queryClient.setQueryData<ToolsetInfo[]>(key, items =>
          items?.map(item => (item.name === 'kanban' ? { ...item, enabled: input.enabled } : item))
        )

        return { previous }
      },
      onError: (error, input, context) => {
        const previous = context?.previous

        if (previous) {
          queryClient.setQueryData<ToolsetInfo[]>([...TOOLSETS_QUERY_KEY, input.scopeKey], items =>
            items?.map(item => (item.name === 'kanban' ? previous : item))
          )
        }

        notifyError(error, p.toggleFailed('Kanban'))
      },
      onSuccess: (_result, input) => {
        notify({ kind: 'success', message: p.kanbanToolsSaved(input.label) })
      },
      onSettled: (_data, _error, input) => {
        // Use the request's identity, never a newly selected foreground profile.
        void queryClient.invalidateQueries({ queryKey: [...TOOLSETS_QUERY_KEY, input.scopeKey], exact: true })
        void queryClient.invalidateQueries({ queryKey: [...SKILLS_QUERY_KEY, input.scopeKey], exact: true })
      }
    },
    queryClient
  )

  const pending = mutation.isPending && mutation.variables?.scopeKey === scopeKey
  const failed = query.isError
  const label = `${p.halfAgentIn(scopeLabel)}: Kanban`

  return (
    <>
      <Switch
        aria-label={label}
        checked={row?.enabled ?? false}
        disabled={!row || failed || query.isFetching || pending}
        onCheckedChange={enabled =>
          mutation.mutate({
            enabled,
            // Pin ambient legacy scopes before onMutate yields to the event loop.
            profile:
              typeof profile === 'object' && profile
                ? { ...profile }
                : {
                    connectionId: getApiRequestConnection(),
                    profile: profile === undefined ? getApiRequestProfile() : profile
                  },
            scopeKey,
            label: scopeLabel
          })
        }
      />
      {failed ? (
        <Tip label={t.skills.toolsetsRefreshFailed}>
          <Button aria-label={t.common.retry} onClick={() => void query.refetch()} size="icon-xs" variant="ghost">
            <Codicon name="refresh" size="0.8rem" />
          </Button>
        </Tip>
      ) : !query.isPending && !row ? (
        <Tip label={p.kanbanToolsUnavailable}>
          <span className="text-[0.65rem] text-(--ui-text-tertiary)">{p.kanbanToolsUpdateBackend}</span>
        </Tip>
      ) : null}
    </>
  )
}
