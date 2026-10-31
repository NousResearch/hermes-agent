/** About ▸ Updates: the source checkout's update channel selector. */
export interface UpdateChannelCopy {
  title: string
  description: string
  stable: string
  main: string
  stableConfirmTitle: string
  stableConfirmBody: string
  stableConfirm: string
  failed: string
  /** Version details: the row naming this install's update channel. */
  detailsLabel: string
  /** A source checkout following a branch other than main. */
  branch: (name: string) => string
  /** Updates overlay: opens Settings › About › Updates, where the channel is chosen. */
  change: string
  /** Up to date on stable: names the release the install is pinned to. */
  latestRelease: (version: string) => string
  /** Updates overlay title: the stable release is older than this checkout (a rollback). */
  rollbackTitle: (version: string) => string
  /** Installing the older stable release moves the checkout back; says so plainly. */
  rollbackBody: (version: string) => string
  /** A stable release `hermes update` refuses: this install keeps its build until a newer release. */
  heldBody: (version: string) => string
  /** Updates overlay install button for that rollback. */
  rollbackAction: (version: string) => string
  /** Update dialog link to every change in a commit-line update. */
  allChanges: string
}
