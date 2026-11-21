/**
 * Kanban output log — no dashboard UI.
 *
 * This plugin contributes a card-drawer section through the Desktop app's
 * `kanban.card.drawer` area; the dashboard web UI adds nothing. The bundle
 * exists only so the manifest's `entry` resolves for dashboard installers that
 * expect one. Registers no slot and no page.
 */
export default {}
