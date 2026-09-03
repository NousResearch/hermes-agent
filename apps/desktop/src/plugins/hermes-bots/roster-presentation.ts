/** Keep group chats behind their own fold even on a single gateway. The
 * gateway-section layout used to be the only path that rendered the group
 * header, so one-device rosters silently lost the collapse affordance. */
export function rosterPresentationMode(
  gatewaySectioned: boolean,
  groupCount: number
): 'flat' | 'gateway-sections' | 'group-section' {
  if (gatewaySectioned) {
    return 'gateway-sections'
  }

  return groupCount > 0 ? 'group-section' : 'flat'
}
