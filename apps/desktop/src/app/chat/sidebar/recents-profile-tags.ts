/**
 * Flat All Profiles recents need per-row ownership unless profile group
 * headers already provide it. Named-profile filtering remains the row's job.
 */
export function showProfileTagsInRecents(showAllProfiles: boolean, groupedByProfile: boolean): boolean {
  return showAllProfiles && !groupedByProfile
}
