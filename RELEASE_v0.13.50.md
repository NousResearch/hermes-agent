## Tasker XML Import Expansion

This release extends Hermes' safe Tasker XML/Data URI import path.

- Imports Tasker Go Home, Back Button, Show Recents, and Quick Settings actions
  as Hermes accessibility global-action automations.
- Imports selected Tasker settings-panel actions as safe Hermes system-action
  automations that open user-facing Android settings screens.
- Keeps imported Tasker records disabled by default and continues to reject
  arbitrary code, plugins, scenes, and unsupported protected mutations.
