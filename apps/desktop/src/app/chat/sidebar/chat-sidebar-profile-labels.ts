import type { SidebarGrouping } from '@/store/layout'
import { ALL_PROFILES } from '@/store/profile'

export interface ProfileLabelFlags {
  /** Show the sidebar's ALL-profiles grouping at all (multi-profile only). */
  showAllProfiles: boolean
  /** Tag each row with its owning profile chip. */
  showProfileTags: boolean
  /** Name the owning profile in each row's metadata line, beside the model. */
  showProfileNames: boolean
}

/**
 * Which profile labels the sidebar's lists carry.
 *
 * Only surface the profile switcher when more than one profile exists, so
 * single-profile users see the unchanged sidebar. Gate ALL-profiles grouping on
 * that too: if a user drops back to one profile while scope is still ALL
 * (persisted), the rail is hidden and they'd otherwise be stuck in the grouped
 * view with no way out.
 *
 * #89888: name the owning profile on every row of a list that merges profiles —
 * the point of Show all is knowing whose chat a row is, and a title plus a model
 * does not say it. Skipped where the list already answers it: profile-grouped
 * lanes carry the name in the group header, and a view narrowed to exactly one
 * profile would repeat the same name on every row.
 */
export function profileLabelFlags(
  profileCount: number,
  profileScope: string,
  grouping: SidebarGrouping,
  profileFilter: readonly string[]
): ProfileLabelFlags {
  const showAllProfiles = profileCount > 1 && profileScope === ALL_PROFILES

  return {
    showAllProfiles,
    showProfileNames: showAllProfiles && grouping !== 'profile' && profileFilter.length !== 1,
    showProfileTags: showAllProfiles
  }
}
